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
    els.compat.textContent =
      "Speech recognition isn't supported in this browser. Please use Chrome or Edge.";
    els.compat.classList.remove("hidden");
    els.micBtn.disabled = true;
    return;
  }
  setupRecognition(SR);
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
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID, message }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
    showAI(data.reply);
    speak(data.reply);
  } catch (err) {
    setState("error", "Something went wrong: " + err.message);
  }
}

/* ---------- Text-to-Speech ---------- */
// Detect Devanagari (Hindi) characters so we can pick a matching voice/lang.
function isHindi(text) {
  return /[ऀ-ॿ]/.test(text);
}

function pickVoice(hindi) {
  const voices = window.speechSynthesis.getVoices();
  if (hindi) {
    const hiPrefer = ["Google हिन्दी", "Lekha", "Kiara", "Microsoft Swara Online"];
    for (const name of hiPrefer) {
      const v = voices.find((x) => x.name === name);
      if (v) return v;
    }
    const hi = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("hi"));
    if (hi) return hi;
    // Fall through to English if no Hindi voice is installed.
  }
  const enPrefer = [
    "Google US English", "Samantha", "Microsoft Aria Online",
    "Microsoft Jenny Online", "Karen", "Daniel",
  ];
  for (const name of enPrefer) {
    const v = voices.find((x) => x.name === name);
    if (v) return v;
  }
  return voices.find((v) => v.lang && v.lang.startsWith("en")) || voices[0] || null;
}

function speak(text) {
  if (!("speechSynthesis" in window)) {
    setState("idle", "Tap the mic to continue");
    return;
  }
  window.speechSynthesis.cancel();
  const hindi = isHindi(text);
  const utter = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(hindi);
  if (voice) utter.voice = voice;
  utter.lang = hindi ? "hi-IN" : "en-US";
  utter.rate = 1.0;
  utter.pitch = 1.0;

  utter.onstart = () => {
    setState("speaking", "Speaking…");
    els.stopSpeak.classList.remove("hidden");
  };
  utter.onend = () => {
    els.stopSpeak.classList.add("hidden");
    setState("idle", "Tap the mic to continue the conversation");
  };
  utter.onerror = () => {
    els.stopSpeak.classList.add("hidden");
    setState("idle", "Tap the mic to continue");
  };
  window.speechSynthesis.speak(utter);
}

/* Some browsers load voices asynchronously. */
if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = () => {};
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
  if (isListening) stopListening();
  else startListening();
});

els.stopSpeak.addEventListener("click", () => {
  window.speechSynthesis && window.speechSynthesis.cancel();
  els.stopSpeak.classList.add("hidden");
  setState("idle", "Tap the mic to continue");
});

boot();
