/* Onboarding wizard (6 screens).
 *
 * - Only authenticated users reach it (auth guard below).
 * - If onboarding_completed === TRUE -> go straight to chat.
 * - Each screen saves immediately to Google Sheets via POST /api/user/profile.
 * - Final screen sets onboarding_completed = TRUE, then goes to chat.
 */
import { getFirebaseAuth } from "./firebase.js";
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";
import { syncUser } from "./userSync.js";

const $ = (id) => document.getElementById(id);
const stepEl = () => $("ob-step");

const state = {
  name: "", age: "", sex: "",
  height_cm: null, weight_kg: null, unit_pref: "ft_kg",
  diet: "", allergies: [], conditions: [],
};

let currentUser = null;
let stepIndex = 0;
const TOTAL = 6;

// The voice onboarding-call service (separate Railway service). After manual
// onboarding we hand off here with the user's Firebase uid so Mira loads their
// profile from the same sheet and greets them by name.
const ONBOARDING_CALL_URL = "https://ai-dietician-production.up.railway.app/";

/* ---------- helpers ---------- */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function setStep(html) { stepEl().innerHTML = html; }
function showErr(msg) {
  const e = $("ob-err");
  if (e) { e.textContent = msg; e.classList.remove("hidden"); }
}
function setProgress() {
  $("ob-bar").style.width = Math.round(((stepIndex + 1) / TOTAL) * 100) + "%";
  $("ob-back").classList.toggle("hidden", stepIndex === 0);
}

/** Save fields to the sheet, update local state, then advance (or finish). */
async function commit(fields, opts = {}) {
  const btn = $("ob-next");
  if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = "Saving…"; }
  try {
    const token = await currentUser.getIdToken();
    const res = await fetch("/api/user/profile", {
      method: "POST",
      headers: { Authorization: "Bearer " + token, "Content-Type": "application/json" },
      body: JSON.stringify({ fields }),
    });
    if (!res.ok) throw new Error((await res.json()).error || "Save failed");
  } catch (err) {
    showErr("Couldn't save — " + err.message);
    if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label || "Next"; }
    return false;
  }
  if (opts.finish) {
    // Manual onboarding complete → go to the voice call (step 2), passing the
    // Firebase uid so the call loads THIS user's profile and greets by name.
    const uid = encodeURIComponent(currentUser.uid);
    window.location.replace(ONBOARDING_CALL_URL + "?uid=" + uid);
    return true;
  }
  stepIndex += 1;
  render();
  return true;
}

/* ---------- Screen 1: Name ---------- */
function screenName() {
  setStep(`
    <h2 class="ob-q">Hi! What should I call you?</h2>
    <input id="ob-name" class="ob-bigfield" type="text" maxlength="30"
           autocomplete="given-name" placeholder="Your name" value="${esc(state.name)}" />
    <p id="ob-err" class="ob-err hidden"></p>
    <button id="ob-next" class="ob-btn">Next</button>
  `);
  const input = $("ob-name");
  input.focus();
  const submit = () => {
    const name = input.value.trim();
    if (!/^[\p{L}\p{M} ]{1,30}$/u.test(name)) {
      showErr("Please enter your name — letters and spaces only.");
      return;
    }
    commit({ name }).then((ok) => { if (ok) state.name = name; });
  };
  $("ob-next").onclick = submit;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); submit(); }
  });
}

/* ---------- Screen 2: Age + Sex ---------- */
function screenAgeSex() {
  const chip = (label, val) =>
    `<button class="ob-chip ${state.sex === val ? "sel" : ""}" data-val="${val}">${label}</button>`;
  setStep(`
    <h2 class="ob-q">A little about you</h2>
    <input id="ob-age" class="ob-bigfield" type="number" inputmode="numeric"
           min="13" max="100" placeholder="Age" value="${esc(state.age)}" />
    <div class="ob-chips ob-row" id="ob-sex">
      ${chip("Male", "male")}${chip("Female", "female")}${chip("Other", "other")}
    </div>
    <p id="ob-err" class="ob-err hidden"></p>
    <button id="ob-next" class="ob-btn">Next</button>
  `);
  const ageEl = $("ob-age");
  ageEl.focus();
  // Max 3 digits.
  ageEl.addEventListener("input", () => {
    if (ageEl.value.length > 3) ageEl.value = ageEl.value.slice(0, 3);
  });

  const validateAge = () => {
    const age = parseInt(ageEl.value, 10);
    if (!ageEl.value || Number.isNaN(age)) { showErr("Please enter your age."); return null; }
    if (age < 13) { showErr("Sorry — Mira is designed for ages 13 and up. 💛"); return null; }
    if (age > 100) { showErr("Please enter an age between 13 and 100."); return null; }
    return age;
  };
  const submit = () => {
    const age = validateAge();
    if (age === null) return;
    if (!state.sex) { showErr("Please pick one."); return; }
    commit({ age, sex: state.sex }).then((ok) => { if (ok) state.age = age; });
  };

  stepEl().querySelectorAll("#ob-sex .ob-chip").forEach((b) => {
    b.onclick = () => {
      state.sex = b.dataset.val;
      stepEl().querySelectorAll("#ob-sex .ob-chip").forEach((x) =>
        x.classList.toggle("sel", x === b));
      $("ob-err").classList.add("hidden");
      // Auto-advance if age is already valid.
      if (ageEl.value && validateAge() !== null) submit();
    };
  });
  $("ob-next").onclick = submit;
}

/* ---------- Screen 3: Height + Weight ---------- */
function screenHeightWeight() {
  let heightMode = state.unit_pref.startsWith("cm") ? "cm" : "ft";
  let weightUnit = state.unit_pref.endsWith("lb") ? "lb" : "kg";

  const ftOpts = (sel) => Array.from({ length: 6 }, (_, i) => i + 3)
    .map((n) => `<option value="${n}" ${sel === n ? "selected" : ""}>${n} ft</option>`).join("");
  const inOpts = (sel) => Array.from({ length: 12 }, (_, i) => i)
    .map((n) => `<option value="${n}" ${sel === n ? "selected" : ""}>${n} in</option>`).join("");

  function render3() {
    const heightBlock = heightMode === "ft"
      ? `<div class="ob-row ob-height">
           <select id="ob-ft" class="ob-select">${ftOpts(5)}</select>
           <select id="ob-in" class="ob-select">${inOpts(8)}</select>
         </div>
         <button id="ob-cmlink" class="ob-link">Prefer cm?</button>`
      : `<input id="ob-cm" class="ob-bigfield" type="number" inputmode="numeric"
                placeholder="Height in cm" min="90" max="275" />
         <button id="ob-ftlink" class="ob-link">Prefer ft / in?</button>`;

    setStep(`
      <h2 class="ob-q">Your height & weight</h2>
      <label class="ob-label">Height</label>
      ${heightBlock}
      <label class="ob-label" style="margin-top:18px">Weight</label>
      <div class="ob-row ob-weight">
        <input id="ob-wt" class="ob-bigfield" type="number" inputmode="decimal"
               placeholder="Weight" />
        <div class="ob-seg" id="ob-wtunit">
          <button data-u="kg" class="${weightUnit === "kg" ? "sel" : ""}">kg</button>
          <button data-u="lb" class="${weightUnit === "lb" ? "sel" : ""}">lb</button>
        </div>
      </div>
      <p id="ob-err" class="ob-err hidden"></p>
      <button id="ob-next" class="ob-btn">Next</button>
    `);

    if ($("ob-cmlink")) $("ob-cmlink").onclick = () => { heightMode = "cm"; render3(); };
    if ($("ob-ftlink")) $("ob-ftlink").onclick = () => { heightMode = "ft"; render3(); };
    stepEl().querySelectorAll("#ob-wtunit button").forEach((b) => {
      b.onclick = () => {
        weightUnit = b.dataset.u;
        stepEl().querySelectorAll("#ob-wtunit button").forEach((x) =>
          x.classList.toggle("sel", x === b));
      };
    });
    $("ob-next").onclick = submit3;
  }

  function submit3() {
    let height_cm;
    if (heightMode === "ft") {
      const ft = parseInt($("ob-ft").value, 10);
      const inch = parseInt($("ob-in").value, 10);
      if (ft < 3 || ft > 8 || inch < 0 || inch > 11) {
        showErr("Please pick a valid height."); return;
      }
      height_cm = Math.round((ft * 12 + inch) * 2.54);
    } else {
      const cm = parseInt($("ob-cm").value, 10);
      if (!cm || cm < 90 || cm > 275) { showErr("Enter height between 90 and 275 cm."); return; }
      height_cm = cm;
    }
    const w = parseFloat($("ob-wt").value);
    if (!w || Number.isNaN(w)) { showErr("Please enter your weight."); return; }
    const weight_kg = weightUnit === "lb"
      ? Math.round(w * 0.453592 * 10) / 10 : Math.round(w * 10) / 10;
    if (weight_kg < 25 || weight_kg > 250) {
      showErr("Please enter a weight between 25 and 250 kg."); return;
    }
    const unit_pref = `${heightMode}_${weightUnit}`;
    try { localStorage.setItem("mira_unit_pref", unit_pref); } catch (_) {}
    commit({ height_cm, weight_kg, unit_pref }).then((ok) => {
      if (ok) { state.height_cm = height_cm; state.weight_kg = weight_kg; state.unit_pref = unit_pref; }
    });
  }

  render3();
}

/* ---------- Screen 4: Diet ---------- */
function screenDiet() {
  const cards = [
    ["vegetarian", "Vegetarian", "no meat, no egg"],
    ["vegan", "Vegan", "no dairy"],
    ["eggetarian", "Eggetarian", "veg + egg"],
    ["non-veg", "Non-veg", "everything"],
    ["jain", "Jain", "no onion, garlic, root veg"],
  ];
  setStep(`
    <h2 class="ob-q">How do you eat?</h2>
    <div class="ob-cards">
      ${cards.map(([v, t, c]) => `
        <button class="ob-card ${state.diet === v ? "sel" : ""}" data-val="${v}">
          <span class="ob-card-t">${t}</span>
          <span class="ob-card-c">${c}</span>
        </button>`).join("")}
    </div>
    <p id="ob-err" class="ob-err hidden"></p>
  `);
  stepEl().querySelectorAll(".ob-card").forEach((b) => {
    b.onclick = () => {
      state.diet = b.dataset.val;
      stepEl().querySelectorAll(".ob-card").forEach((x) =>
        x.classList.toggle("sel", x === b));
      commit({ diet: b.dataset.val });  // tap = select + auto-advance
    };
  });
}

/* ---------- Screens 5 & 6: multi-select + None + free text ---------- */
function multiSelectScreen({ title, field, options, freePlaceholder, footer, extraChip, finish }) {
  const selected = new Set(state[field] || []);
  let noneSel = Array.isArray(state[field]) && state[field].length === 0 && state[`_${field}_none`];

  const chips = [...options];
  if (extraChip) chips.push(extraChip);

  setStep(`
    <h2 class="ob-q">${title}</h2>
    <div class="ob-chips ob-grid" id="ms-chips">
      ${chips.map((o) => `<button class="ob-chip" data-val="${esc(o)}">${esc(o)}</button>`).join("")}
    </div>
    <button id="ms-none" class="ob-none">None</button>
    <input id="ms-free" class="ob-freetext" type="text" placeholder="${esc(freePlaceholder)}" />
    <div class="ob-chips ob-grid" id="ms-typed"></div>
    ${footer ? `<p class="ob-footer">${footer}</p>` : ""}
    <p id="ob-err" class="ob-err hidden"></p>
    <button id="ob-next" class="ob-btn">Next</button>
  `);

  const chipsWrap = $("ms-chips");
  const noneBtn = $("ms-none");
  const freeEl = $("ms-free");
  const typedWrap = $("ms-typed");

  function renderTyped() {
    const items = typedItems();
    typedWrap.innerHTML = items
      .map((t) => `<span class="ob-chip typed">${esc(t)}</span>`).join("");
  }
  function typedItems() {
    return freeEl.value.split(",").map((s) => s.trim()).filter(Boolean);
  }
  function syncNone() { noneBtn.classList.toggle("sel", noneSel); }

  chipsWrap.querySelectorAll(".ob-chip").forEach((b) => {
    if (selected.has(b.dataset.val)) b.classList.add("sel");
    b.onclick = () => {
      noneSel = false; syncNone();           // picking a chip clears None
      if (selected.has(b.dataset.val)) { selected.delete(b.dataset.val); b.classList.remove("sel"); }
      else { selected.add(b.dataset.val); b.classList.add("sel"); }
    };
  });
  noneBtn.onclick = () => {
    noneSel = true; syncNone();
    selected.clear();                          // None clears all chips
    chipsWrap.querySelectorAll(".ob-chip").forEach((x) => x.classList.remove("sel"));
  };
  freeEl.addEventListener("input", () => {
    if (typedItems().length) { noneSel = false; syncNone(); }
    renderTyped();
  });

  $("ob-next").onclick = () => {
    const typed = typedItems();
    // On the final screen, mark onboarding complete in the same save.
    const finishFields = finish ? { onboarding_completed: "TRUE" } : {};
    if (noneSel) {
      commit({ [field]: [], ...finishFields }, { finish }).then((ok) => {
        if (ok) { state[field] = []; state[`_${field}_none`] = true; }
      });
      return;
    }
    const combined = [...new Set([...selected, ...typed].map((s) => s.toLowerCase()))];
    if (combined.length === 0) {
      showErr("Please pick at least one, choose None, or type something.");
      return;
    }
    commit({ [field]: combined, ...finishFields }, { finish }).then((ok) => {
      if (ok) { state[field] = combined; state[`_${field}_none`] = false; }
    });
  };
}

function screenAllergies() {
  multiSelectScreen({
    title: "Any food allergies?",
    field: "allergies",
    options: ["peanut", "tree nuts", "dairy", "gluten", "soy", "egg", "shellfish"],
    freePlaceholder: "Anything else? e.g. brinjal, sesame",
    finish: false,
  });
}

function screenConditions() {
  const extra = state.sex === "female" ? "pregnant / breastfeeding" : null;
  multiSelectScreen({
    title: "Any health conditions?",
    field: "conditions",
    options: ["diabetes", "PCOS/PCOD", "thyroid", "high BP", "cholesterol", "IBS/acidity", "kidney issue"],
    freePlaceholder: "Anything else? e.g. migraine, fatty liver",
    extraChip: extra,
    footer: "This stays private and helps Mira keep advice safe for you.",
    finish: true,   // last screen -> completes onboarding
  });
}

/* ---------- router ---------- */
const SCREENS = [
  screenName, screenAgeSex, screenHeightWeight,
  screenDiet, screenAllergies, screenConditions,
];

function render() {
  setProgress();
  SCREENS[stepIndex]();
}

function goBack() {
  if (stepIndex > 0) { stepIndex -= 1; render(); }
}

/* ---------- boot: auth gate ---------- */
(async () => {
  let auth;
  try {
    auth = await getFirebaseAuth();
  } catch (_) {
    window.location.replace("/login");
    return;
  }
  onAuthStateChanged(auth, async (user) => {
    if (!user) { window.location.replace("/login"); return; }
    currentUser = user;
    const info = await syncUser(user);
    const done = info && String(info.onboarding_completed).toUpperCase() === "TRUE";
    if (done) { window.location.replace("/"); return; }
    // Show the wizard.
    $("ob-loading").classList.add("hidden");
    $("ob-app").classList.remove("hidden");
    $("ob-back").onclick = goBack;
    render();
  });
})();
