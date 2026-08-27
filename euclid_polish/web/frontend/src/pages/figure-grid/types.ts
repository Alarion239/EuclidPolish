export type FigureTier = "dirty" | "sr" | "hr" | "bhr" | "jwst";
export type FigureMode = "VIS" | "H_E" | "VIS_H" | "native";
export type FigureRegime = "real" | "synthetic";
export type FigureRecipeKey = `${FigureTier}:${FigureMode}`;

export type FigureRecipe = {
  tier: FigureTier;
  mode: FigureMode;
  title: string;
};

export type RecipeRow = FigureRecipe & { id: string };

export type SavedResultSource = {
  collection?: string;
  index?: number;
  object?: { label?: string };
};

export type SavedResult = {
  id: string;
  label: string;
  regime?: FigureRegime;
  source?: SavedResultSource;
  selection?: Record<string, unknown>;
  logical_tiers: string[];
  bands: Record<string, string[]>;
  pixscale_arcsec: Record<string, number>;
  recipes: FigureRecipeKey[];
  wcs_preserved: boolean;
};

export type ViewerResultsIndex = {
  schema_version?: number;
  limits?: {
    max_results?: number;
    max_rows?: number;
  };
  axis_defaults?: {
    columns?: string;
    rows?: string;
  };
  supported?: {
    logical_tiers?: string[];
    modes?: string[];
  };
  results: SavedResult[];
};
