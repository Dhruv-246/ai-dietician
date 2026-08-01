/* Signup page: create an email/password account, then go to the chat page. */
import { getFirebaseAuth } from "./firebase.js";
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";
import { friendlyAuthError } from "./authErrors.js";
import { syncUser } from "./userSync.js";

const form = document.getElementById("signup-form");
const emailEl = document.getElementById("email");
const passwordEl = document.getElementById("password");
const errorEl = document.getElementById("error");
const submitBtn = document.getElementById("submit");

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

let auth;
try {
  auth = await getFirebaseAuth();
} catch (err) {
  showError(friendlyAuthError(err));
  submitBtn.disabled = true;
}

// Already logged in? Go to the chat page.
if (auth) {
  onAuthStateChanged(auth, (user) => {
    if (user) window.location.replace("/");
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.classList.add("hidden");
  if (!auth) return;
  submitBtn.disabled = true;
  submitBtn.textContent = "Creating account…";
  try {
    const cred = await createUserWithEmailAndPassword(
      auth, emailEl.value.trim(), passwordEl.value
    );
    // Create the Users-sheet row, then send the new user into onboarding.
    await syncUser(cred.user);
    window.location.replace("/onboarding");
  } catch (err) {
    showError(friendlyAuthError(err));
    submitBtn.disabled = false;
    submitBtn.textContent = "Sign up";
  }
});
