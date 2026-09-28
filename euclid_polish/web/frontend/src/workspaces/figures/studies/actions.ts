/* The explicit study writes shared by the list and the study view: resume an
 * incomplete study and delete one (confirmed; a study with fields also loses
 * its holylabs store). Each asks first or is a button press; none runs on a
 * page open. */
import { ApiError, apiPost } from "../../../api/client";
import { refreshJobsFeed, useJobsStore } from "../../../api/jobs";
import { invalidate } from "../../../api/query";
import { confirm, toast } from "../../../ui";
import { FREEZE_JOB_KEY, refusalMessage, type FreezeReply } from "../../shared/studyFreeze";
import { STUDIES_URL, studyUrl } from "./api";

/** POST resume → the freeze job (re-attached under the freeze key), or a toast with the refusal. */
export async function resumeStudy(id: string): Promise<string | null> {
  try {
    const r = await apiPost<FreezeReply>(`${studyUrl(id)}/resume`, {});
    if (!r.ok || !r.job_id) { toast.error(r.error ?? "The resume did not start"); return null; }
    useJobsStore.getState().register(r.job_id, FREEZE_JOB_KEY);
    void refreshJobsFeed();
    invalidate(STUDIES_URL);
    return r.job_id;
  } catch (e) {
    const m = refusalMessage(e);
    toast.error(`${m.title}: ${m.text}`);
    return null;
  }
}

/** Confirm, then delete; offline with fields asks again to delete only the local copy. */
export async function deleteStudy(study: { id: string; name: string; fields: number }): Promise<boolean> {
  const ok = await confirm({
    title: `Delete the study “${study.name}”?`,
    message: study.fields
      ? `Its numbers here and its ${study.fields} attached field${study.fields === 1 ? "" : "s"} on holylabs are deleted. This cannot be undone.`
      : "Its numbers here (and their holylabs mirror) are deleted. This cannot be undone.",
    tone: "danger", confirmLabel: "Delete",
  });
  if (!ok) return false;
  const send = (localOnly: boolean) => apiPost<{ ok: boolean; error?: string }>(`${studyUrl(study.id)}/delete`, { confirm: 1, local_only: localOnly ? 1 : null });
  try {
    await send(false);
  } catch (e) {
    if (e instanceof ApiError && (e.status === 503 || (e.status === 409 && e.code !== "busy"))) {
      const local = await confirm({
        title: "Delete only the local copy?",
        message: `${e.message}. The holylabs copy of its fields then stays (delete it later when FASRC is connected).`,
        tone: "danger", confirmLabel: "Delete local copy",
      });
      if (!local) return false;
      try { await send(true); } catch (e2) { toast.error(e2 instanceof Error ? e2.message : String(e2)); return false; }
    } else {
      toast.error(e instanceof Error ? e.message : String(e));
      return false;
    }
  }
  toast.success(`Deleted the study “${study.name}”`);
  invalidate(STUDIES_URL);
  return true;
}
