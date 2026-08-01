/* Chat-page auth guard.
 *
 * Protects the chat page: if no user is signed in, redirect to /login.
 * When a user IS signed in, reveal the app and wire the logout button.
 * Does not touch any chat logic (app.js handles that independently).
 */
import { getFirebaseAuth } from "./firebase.js";
import {
  onAuthStateChanged,
  signOut,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";
import { syncUser } from "./userSync.js";

function redirectToLogin() {
  window.location.replace("/login");
}

let auth;
try {
  auth = await getFirebaseAuth();
} catch (err) {
  // Not configured -> send to login, which surfaces the message.
  redirectToLogin();
}

if (auth) {
  onAuthStateChanged(auth, async (user) => {
    if (!user) {
      redirectToLogin();
      return;
    }
    // Ensure a Users-sheet row exists and learn onboarding status.
    const info = await syncUser(user);
    const done =
      info && String(info.onboarding_completed).toUpperCase() === "TRUE";
    if (!done) {
      window.location.replace("/onboarding");
      return;
    }
    // Expose a token getter so app.js can authenticate its /api/chat calls.
    window.__miraGetToken = () => user.getIdToken();
    // Authenticated + onboarded: reveal the chat UI.
    document.body.classList.add("authed");
  });

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      await signOut(auth);
      redirectToLogin();
    });
  }
}
