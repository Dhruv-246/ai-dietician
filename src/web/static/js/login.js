/* Login page: sign in with email/password, then go to the chat page.
 * If a user is already signed in, skip straight to the chat page.
 */
import { getFirebaseAuth } from "./firebase.js";
import {
  signInWithEmailAndPassword,
  onAuthStateChanged,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";
import { friendlyAuthError } from "./authErrors.js";
import { syncUser } from "./userSync.js";
import { landingTarget } from "./route.js";

const form = document.getElementById("login-form");
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

// Already logged in? Route to the right step (onboarding / onboarding call / Step-3 call).
if (auth) {
  onAuthStateChanged(auth, async (user) => {
    if (!user) return;
    try {
      const info = await syncUser(user);
      window.location.replace(landingTarget(info, user.uid));
    } catch (_) {
      window.location.replace("/onboarding");
    }
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.classList.add("hidden");
  if (!auth) return;
  submitBtn.disabled = true;
  submitBtn.textContent = "Logging in…";
  try {
    const cred = await signInWithEmailAndPassword(
      auth, emailEl.value.trim(), passwordEl.value
    );
    // Ensure the Users-sheet row exists (idempotent) and route by onboarding status.
    const info = await syncUser(cred.user);
    window.location.replace(landingTarget(info, cred.user.uid));
  } catch (err) {
    showError(friendlyAuthError(err));
    submitBtn.disabled = false;
    submitBtn.textContent = "Log in";
  }
});
