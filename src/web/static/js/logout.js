/* Logout page: sign the user out of Firebase, then send them to /login.
 * Reached from the Step-3 call screen's "Log out" link (which lives on the
 * agent origin and can't sign out itself — Firebase auth lives here on the web
 * origin, so logout happens on this page). */
import { getFirebaseAuth } from "./firebase.js";
import { signOut } from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";

try {
  const auth = await getFirebaseAuth();
  await signOut(auth);
} catch (_) {
  // Even if sign-out fails, send them to login.
}
window.location.replace("/login");
