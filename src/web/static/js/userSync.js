/* Syncs the signed-in Firebase user into the Users sheet.
 *
 * Sends the Firebase ID token to the backend, which verifies it and
 * get-or-creates the Users row. Idempotent — safe to call on every login.
 */
export async function syncUser(user) {
  if (!user) return null;
  try {
    const token = await user.getIdToken();
    const res = await fetch("/api/user/sync", {
      method: "POST",
      headers: { Authorization: "Bearer " + token },
    });
    if (!res.ok) {
      console.error("user sync failed:", await res.text());
      return null;
    }
    return await res.json();
  } catch (err) {
    console.error("user sync error:", err);
    return null;
  }
}
