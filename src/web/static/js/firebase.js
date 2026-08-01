/* Firebase initialization (shared by login, signup, and the chat guard).
 *
 * Loads the Firebase web config from the backend (/api/firebase-config), which
 * reads it from environment variables. The config values are public by design.
 * Persistence is set to LOCAL so users stay logged in across refreshes.
 */
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.5/firebase-app.js";
import {
  getAuth,
  setPersistence,
  browserLocalPersistence,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";

let _authPromise = null;

/** Returns a ready Firebase Auth instance (memoized). Throws if unconfigured. */
export function getFirebaseAuth() {
  if (!_authPromise) {
    _authPromise = (async () => {
      const res = await fetch("/api/firebase-config");
      const cfg = await res.json();
      if (!cfg || !cfg.apiKey) {
        throw new Error(
          "Firebase is not configured. Set the FIREBASE_* environment variables."
        );
      }
      const app = initializeApp(cfg);
      const auth = getAuth(app);
      // Keep the user signed in across page refreshes and tabs.
      await setPersistence(auth, browserLocalPersistence);
      return auth;
    })();
  }
  return _authPromise;
}
