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
import { landingTarget } from "./route.js";

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
    // Step 3 is voice-only: "/" is a pure router, never the chat screen.
    // Send the user to their current step — onboarding form, onboarding call,
    // or the ongoing "talk to Mira" call — based on their progress.
    let info = null;
    try {
      info = await syncUser(user);
    } catch (_) {}
    window.location.replace(landingTarget(info, user.uid));
  });

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      await signOut(auth);
      redirectToLogin();
    });
  }
}
