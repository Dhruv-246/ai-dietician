/* AI Dietician — voice-to-voice frontend.
 *
 * Responsibilities (audio I/O only; no backend logic here):
 *   1. Capture speech via the Web Speech API (SpeechRecognition).
 *   2. Show the transcript, POST it to /api/chat.
 *   3. Speak the reply via speechSynthesis (TTS) and show its text.
 *
 * State machine: idle -> listening -> thinking -> speaking -> idle
 */

const els = {
  aiName: document.getElementById("ai-name"),
  orb: document.getElementById("orb"),
  status: document.getElementById("status"),
  micBtn: document.getElementById("mic-btn"),
  userBubble: document.getElementById("user-bubble"),
  userText: document.getElementById("user-text"),
  aiBubble: document.getElementById("ai-bubble"),
  aiText: document.getElementById("ai-text"),
  stopSpeak: document.getElementById("stop-speak"),
  compat: document.getElementById("compat-note"),
  textForm: document.getElementById("text-form"),
  textInput: document.getElementById("text-input"),
};

let USER_ID = "U001";
let recognition = null;
let isListening = false;
let state = "idle";

/* ---------- UI state ---------- */
function setState(next, message) {
  state = next;
  els.orb.className = "orb state-" + next;
  els.status.className = "status " + (next === "idle" ? "" : next);
  if (message) els.status.textContent = message;
}

function showUser(text) {
  els.userText.textContent = text;
  els.userBubble.classList.remove("hidden");
}
function showAI(text) {
  els.aiText.textContent = text;
  els.aiBubble.classList.remove("hidden");
}

/* ---------- Bootstrap ---------- */
async function boot() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    USER_ID = cfg.user_id || USER_ID;
    if (cfg.ai_name) els.aiName.textContent = cfg.ai_name;
  } catch (_) { /* keep defaults */ }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    // iPhone/Safari: no mic speech-to-text. Guide the user to type instead.
    els.compat.textContent =
      "Voice input isn't supported on this device (e.g. iPhone). Type below — " +
      "or tap the mic on your keyboard to dictate.";
    els.compat.classList.remove("hidden");
    els.micBtn.disabled = true;
    els.status.textContent = "Type your message to start";
    if (els.textInput) els.textInput.focus();
    return;
  }
  setupRecognition(SR);
}

/* iOS blocks speechSynthesis unless first triggered inside a user gesture.
 * Warm it up on the user's tap so the async reply can be spoken later. */
let ttsWarmed = false;
function warmUpTTS() {
  if (ttsWarmed || !("speechSynthesis" in window)) return;
  try {
    const u = new SpeechSynthesisUtterance(" ");
    u.volume = 0;
    window.speechSynthesis.speak(u);
  } catch (_) { /* ignore */ }
  ttsWarmed = true;
}

/* ---------- Speech-to-Text ---------- */
function setupRecognition(SR) {
  recognition = new SR();
  // hi-IN recognizes Hindi and handles common Hindi+English (Hinglish) speech.
  recognition.lang = "hi-IN";
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.maxAlternatives = 1;

  let finalTranscript = "";

  recognition.onstart = () => {
    isListening = true;
    finalTranscript = "";
    els.micBtn.classList.add("recording");
    setState("listening", "Listening…");
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const chunk = event.results[i][0].transcript;
      if (event.results[i].isFinal) finalTranscript += chunk;
      else interim += chunk;
    }
    const shown = (finalTranscript + interim).trim();
    if (shown) showUser(shown);
  };

  recognition.onerror = (event) => {
    isListening = false;
    els.micBtn.classList.remove("recording");
    if (event.error === "no-speech") {
      setState("idle", "I didn't catch that — tap the mic to try again.");
    } else if (event.error === "not-allowed") {
      setState("error", "Microphone permission denied. Allow mic access and retry.");
    } else {
      setState("error", "Mic error: " + event.error);
    }
  };

  recognition.onend = () => {
    isListening = false;
    els.micBtn.classList.remove("recording");
    const text = finalTranscript.trim();
    if (text) {
      showUser(text);
      sendToBackend(text);
    } else if (state === "listening") {
      setState("idle", "Tap the mic to start talking");
    }
  };
}

/* ---------- Backend call ---------- */
async function sendToBackend(message) {
  setState("thinking", "Thinking…");
  els.aiBubble.classList.add("hidden");
  try {
    // Identify the user by their Firebase token (set by guard.js), not a
    // client-supplied id — the backend resolves the row from firebase_uid.
    if (!window.__miraGetToken) throw new Error("not signed in");
    const token = await window.__miraGetToken();
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,
      },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
    showAI(data.reply);
    speak(data.reply);
    // Background: extract long-term memory from this exchange. Non-blocking —
    // does not affect the chat reply or voice playback.
    fetch("/api/memory/extract", {
      method: "POST",
      headers: { Authorization: "Bearer " + token },
    }).catch(() => {});
  } catch (err) {
    setState("error", "Something went wrong: " + err.message);
  }
}

/* ---------- Text-to-Speech ---------- */
// Detect Devanagari (Hindi) characters so we can pick a matching voice/lang.
function isHindi(text) {
  return /[ऀ-ॿ]/.test(text);
}

/* Voice cache — getVoices() is empty on first call in Chrome and only fills
 * after the 'voiceschanged' event. We cache voices there so pickVoice() never
 * silently falls back to the default US voice due to that race. */
let voicesCache = [];
function loadVoices() {
  if (!("speechSynthesis" in window)) return;
  const v = window.speechSynthesis.getVoices();
  if (v && v.length) voicesCache = v;
}
if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = loadVoices;
  loadVoices();
}

/* Format the LLM reply for speech (not for reading):
 * strip markdown, remove emoji, expand units to words, soften dashes. */
function formatForSpeech(text) {
  let t = String(text || "");
  // Remove emoji / pictographs / variation selectors.
  t = t.replace(
    /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{2190}-\u{21FF}\u{FE0F}\u{200D}]/gu,
    ""
  );
  // Markdown: code fences/inline, bold/italic markers, headings, quotes, links, rules.
  t = t.replace(/```[\s\S]*?```/g, " ").replace(/`([^`]*)`/g, "$1");
  t = t.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\*(.*?)\*/g, "$1");
  t = t.replace(/__(.*?)__/g, "$1").replace(/_(.*?)_/g, "$1");
  t = t.replace(/^#{1,6}\s+/gm, "").replace(/^\s*>\s?/gm, "");
  t = t.replace(/^\s*[-*•]\s+/gm, "");          // list bullets
  t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1"); // [text](url) -> text
  t = t.replace(/^\s*[-–—]{3,}\s*$/gm, " ");     // horizontal rules
  // Number ranges "10–20" -> "10 to 20" (before dash softening).
  t = t.replace(/(\d)\s*[–—-]\s*(\d)/g, "$1 to $2");
  // Units -> spoken words (kg/mg before g; keep the digit for the voice to read).
  t = t.replace(/(\d+(?:\.\d+)?)\s*kcal\b/gi, "$1 calories");
  t = t.replace(/(\d+(?:\.\d+)?)\s*cal\b/gi, "$1 calories");
  t = t.replace(/(\d+(?:\.\d+)?)\s*kg\b/gi, "$1 kilograms");
  t = t.replace(/(\d+(?:\.\d+)?)\s*mg\b/gi, "$1 milligrams");
  t = t.replace(/(\d+(?:\.\d+)?)\s*g\b/gi, "$1 grams");
  t = t.replace(/(\d+(?:\.\d+)?)\s*ml\b/gi, "$1 millilitre");
  t = t.replace(/(\d+(?:\.\d+)?)\s*[lL]\b/g, "$1 litre");
  t = t.replace(/(\d+(?:\.\d+)?)\s*min\b/gi, "$1 minutes");
  t = t.replace(/(\d+(?:\.\d+)?)\s*hrs?\b/gi, "$1 hours");
  // Clause dashes (spaced) -> comma pause; newlines -> sentence breaks.
  t = t.replace(/\s+[–—-]\s+/g, ", ");
  t = t.replace(/\s*\n+\s*/g, ". ");
  // Tidy whitespace and collapse repeated/stray punctuation.
  t = t.replace(/\s+([,.!?])/g, "$1");
  t = t.replace(/([,.!?])(?:\s*[,.!?])+/g, "$1 ");
  t = t.replace(/\s{2,}/g, " ").trim();
  return t;
}

function pickVoice(hindi) {
  const voices = (voicesCache && voicesCache.length)
    ? voicesCache
    : (("speechSynthesis" in window && window.speechSynthesis.getVoices()) || []);
  const byName = (names) => {
    for (const n of names) {
      const v = voices.find((x) => x.name === n);
      if (v) return v;
    }
    return null;
  };
  const byLang = (prefix) =>
    voices.find((v) => v.lang && v.lang.toLowerCase().startsWith(prefix)) || null;

  if (hindi) {
    return byName(["Google हिन्दी", "Microsoft Swara Online", "Lekha", "Kiara"])
      || byLang("hi")
      // No Hindi voice installed -> still prefer an Indian-accent voice.
      || byName(["Rishi"]) || byLang("en-in")
      || voices[0] || null;
  }
  // English: Indian English first, then UK, then any non-US en; US voice is last.
  return byName(["Rishi", "Heera", "Ravi"])
    || byLang("en-in")
    || byName(["Google UK English"]) || byLang("en-gb")
    || voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("en")
                          && v.lang.toLowerCase() !== "en-us")
    || byLang("en")
    || voices[0] || null;
}

/* TTS PROVIDER SEAM — the only browser-specific speech code lives here.
 * To move TTS server-side later, replace the body of speakWithProvider() with a
 * fetch to your TTS endpoint (send `text`, play the returned audio, call the
 * same handlers). Nothing else in app.js needs to change. */
function speakWithProvider(text, hindi, handlers) {
  const utter = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(hindi);
  if (voice) utter.voice = voice;
  utter.lang = hindi ? "hi-IN" : (voice && voice.lang ? voice.lang : "en-IN");
  utter.rate = 0.95; // 1.0 is slightly fast for code-mixed speech
  utter.pitch = 1.0;
  // Debug: shows exactly which voice/lang is used for each utterance.
  console.log(
    `[Mira TTS] voice="${voice ? voice.name : "(default)"}" lang="${utter.lang}" hindi=${hindi}`
  );
  utter.onstart = handlers.onstart;
  utter.onend = handlers.onend;
  utter.onerror = handlers.onerror;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

/* Handoff: sanitize the reply, choose script, hand to the TTS provider. */
function speak(text) {
  if (!("speechSynthesis" in window)) {
    setState("idle", "Tap the mic to continue");
    return;
  }
  const spoken = formatForSpeech(text);
  const hindi = isHindi(spoken);
  speakWithProvider(spoken, hindi, {
    onstart: () => {
      setState("speaking", "Speaking…");
      els.stopSpeak.classList.remove("hidden");
    },
    onend: () => {
      els.stopSpeak.classList.add("hidden");
      setState("idle", "Tap the mic to continue the conversation");
    },
    onerror: () => {
      els.stopSpeak.classList.add("hidden");
      setState("idle", "Tap the mic to continue");
    },
  });
}

/* ---------- Controls ---------- */
function startListening() {
  if (!recognition) return;
  window.speechSynthesis && window.speechSynthesis.cancel();
  els.stopSpeak.classList.add("hidden");
  try {
    recognition.start();
  } catch (_) { /* start() throws if already started; ignore */ }
}

function stopListening() {
  if (recognition && isListening) recognition.stop();
}

els.micBtn.addEventListener("click", () => {
  if (state === "thinking") return; // ignore taps while waiting on the LLM
  warmUpTTS(); // unlock iOS/Safari audio within this gesture
  if (isListening) stopListening();
  else startListening();
});

/* Text input fallback — the primary path on iPhone. */
if (els.textForm) {
  els.textForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state === "thinking") return;
    const msg = (els.textInput.value || "").trim();
    if (!msg) return;
    warmUpTTS(); // unlock TTS within this user gesture
    els.textInput.value = "";
    showUser(msg);
    sendToBackend(msg);
  });
}

els.stopSpeak.addEventListener("click", () => {
  window.speechSynthesis && window.speechSynthesis.cancel();
  els.stopSpeak.classList.add("hidden");
  setState("idle", "Tap the mic to continue");
});

boot();
