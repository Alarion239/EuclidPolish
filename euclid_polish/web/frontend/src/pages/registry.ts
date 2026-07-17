/* Single source of truth for SPA pages. Each entry drives BOTH the sidebar nav
   and the router — add a page here and it's wired everywhere (DRY). The full
   console is now ported; every tab routes in the SPA. */
import type { ComponentType } from "react";
import CatalogPage from "./Catalog";
import ConfigPage from "./Config";
import CutoutsPage from "./Cutouts";
import EnsemblePage from "./Ensemble";
import EvaluationPage from "./Evaluation";
import FasrcPage from "./Fasrc";
import GitPage from "./Git";
import InferencePage from "./Inference";
import LensFinderPage from "./LensFinder";
import LensIsolationPage from "./LensIsolation";
import PsfsPage from "./Psfs";
import SkyPage from "./Sky";
import SyntheticRealPage from "./SyntheticReal";
import TngPage from "./Tng";
import TrackingPage from "./Tracking";
import TrainMembersPage from "./TrainMembers";
import VisualizationPage from "./Visualization";

export type PageDef = { label: string; path: string; component: ComponentType };
export type NavGroup = { title: string; items: { label: string; path: string }[] };

/** Every routed SPA page, keyed by path — this list IS the router + the nav. */
export const PAGES: PageDef[] = [
  { label: "Config", path: "/config", component: ConfigPage },
  { label: "Catalog", path: "/catalog", component: CatalogPage },
  { label: "PSFs", path: "/psfs", component: PsfsPage },
  { label: "Sky", path: "/sky", component: SkyPage },
  { label: "Cutouts", path: "/cutouts", component: CutoutsPage },
  { label: "TNG", path: "/tng", component: TngPage },
  { label: "Synthetic–Real", path: "/synthetic-real", component: SyntheticRealPage },
  { label: "Inference", path: "/inference", component: InferencePage },
  { label: "Ensemble", path: "/ensemble", component: EnsemblePage },
  { label: "Train members", path: "/train-members", component: TrainMembersPage },
  { label: "Evaluation", path: "/evaluation", component: EvaluationPage },
  { label: "Lens finder", path: "/lensfinder", component: LensFinderPage },
  { label: "Lens isolation", path: "/lens-isolation", component: LensIsolationPage },
  { label: "Tracking", path: "/tracking", component: TrackingPage },
  { label: "Visualization", path: "/visualization", component: VisualizationPage },
  { label: "FASRC", path: "/fasrc", component: FasrcPage },
  { label: "Git", path: "/git", component: GitPage },
];

const path = (label: string) => PAGES.find((p) => p.label === label)!.path;
const item = (label: string) => ({ label, path: path(label) });

/** Sidebar sections in the reading order of a run: setup → data → model → ops. */
export const NAV: NavGroup[] = [
  { title: "Setup", items: [item("Config"), item("Catalog"), item("PSFs")] },
  { title: "Data", items: [item("Sky"), item("Cutouts"), item("TNG"), item("Synthetic–Real")] },
  { title: "Model", items: [item("Inference"), item("Ensemble"), item("Train members"), item("Evaluation"), item("Lens finder"), item("Lens isolation")] },
  { title: "Ops", items: [item("Tracking"), item("Visualization"), item("FASRC"), item("Git")] },
];
