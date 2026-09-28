/* Figures › Studies (`/figures/studies`, spec 2026-09-28 "Model studies"):
 * frozen whole-ensemble comparisons for the paper. Without `?study=` the
 * list of studies (open, resume an incomplete one, delete — confirmed); with
 * `?study=<id>` one study: its selection, the six charts drawn from its
 * frozen numbers with their exports, and its attached fields. Opening the
 * tab or a study reads only: it never starts a job and never fetches a
 * field. */
import { useUrlState } from "../../../hooks/useUrlState";
import { Page } from "../../../ui";
import { StudyList } from "../studies/StudyList";
import { StudyView } from "../studies/StudyView";
import "../figures.css";
import "../studies/studies.css";

export default function Studies() {
  const [study] = useUrlState("study", "");
  return (
    <Page className="fig-page stu-page">
      {study ? <StudyView id={study} /> : <StudyList />}
    </Page>
  );
}
