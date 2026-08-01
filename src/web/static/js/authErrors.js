/* Maps Firebase Auth error codes to friendly, user-facing messages. */
export function friendlyAuthError(err) {
  const code = (err && err.code) || "";
  const map = {
    "auth/invalid-email": "That email address doesn't look right.",
    "auth/missing-password": "Please enter your password.",
    "auth/weak-password": "Password should be at least 6 characters.",
    "auth/email-already-in-use": "An account with this email already exists. Try logging in.",
    "auth/invalid-credential": "Incorrect email or password.",
    "auth/wrong-password": "Incorrect email or password.",
    "auth/user-not-found": "No account found with this email — sign up first.",
    "auth/too-many-requests": "Too many attempts. Please try again in a moment.",
    "auth/network-request-failed": "Network error. Check your connection and retry.",
    // Firebase project configuration problems:
    "auth/operation-not-allowed":
      "Email/Password sign-in isn't enabled for this project. Enable it in Firebase Console → Authentication → Sign-in method.",
    "auth/configuration-not-found":
      "Firebase Authentication isn't set up yet. In Firebase Console → Authentication, click Get started and enable Email/Password.",
    "auth/unauthorized-domain":
      "This domain isn't authorized in Firebase → Authentication → Settings → Authorized domains.",
    "auth/api-key-not-valid": "The Firebase API key looks invalid.",
    "auth/invalid-api-key": "The Firebase API key looks invalid.",
  };
  if (map[code]) return map[code];
  if (err && err.message && err.message.includes("not configured")) {
    return err.message;
  }
  // Never hide the code again — surface it so problems are diagnosable.
  return code
    ? `Something went wrong (${code}). Please try again.`
    : "Something went wrong. Please try again.";
}
