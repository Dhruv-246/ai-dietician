/* Post-auth landing router.
 *
 * The product has three steps and this decides where a signed-in user lands:
 *   Step 1 not done            -> manual onboarding form
 *   Step 1 done, Step 2 not    -> onboarding CALL (Mira leads)          [agent, no mode]
 *   Steps 1 & 2 done           -> ongoing "ask Mira" CALL (Step 3)      [agent, mode=ongoing]
 *
 * `info` is the object returned by /api/user/sync (onboarding_completed +
 * onboarding_call_done). `uid` is the Firebase uid, handed to the call so it
 * loads that user's profile + long-term memory.
 */
export const AGENT_URL = "https://ai-dietician-production.up.railway.app/";

export function landingTarget(info, uid) {
  const yes = (v) => info && String(v).toUpperCase() === "TRUE";
  const completed = yes(info && info.onboarding_completed);
  const callDone = yes(info && info.onboarding_call_done);
  if (!completed) return "/onboarding";
  const u = encodeURIComponent(uid || "");
  if (!callDone) return AGENT_URL + "?uid=" + u; // Step 2: onboarding call
  return AGENT_URL + "?uid=" + u + "&mode=ongoing"; // Step 3: ask Mira
}
