/* figures/grid — legacy adapter: the figure-grid builder that heads the old
   Visualization page (plates live in figures/plates). */
import FigureGridBuilder from "../../../pages/figure-grid/FigureGridBuilder";
import { Page, PageHead } from "../../../ui";

export default function Grid() {
  return (
    <Page>
      <PageHead eyebrow="figures · grid" title="Figure grid"
        sub="Assemble saved viewer crops into a publication grid." />
      <FigureGridBuilder />
    </Page>
  );
}
