/* What a <Field> tells the kit controls inside it (ui/controls.tsx): the id
   and plain text of its label (for controls that name themselves with
   aria-labelledby), the ids that describe the control (description, then
   error) and the invalid flag. Overlays (Popover, Dialog) reset it to null,
   because React context crosses portals: a control inside a popover opened
   from within a Field is not that Field's control. */
import { createContext, useContext, type AriaAttributes } from "react";

export type FieldCtx = { labelId: string; labelText?: string; describedBy?: string; invalid: boolean };

export const FieldContext = createContext<FieldCtx | null>(null);

export function useField(): FieldCtx | null {
  return useContext(FieldContext);
}

/** aria-describedby / aria-invalid for a control inside a Field: the caller's
 *  own description first, then the Field's description and error. */
export function useFieldAria(describedBy?: string, invalid?: AriaAttributes["aria-invalid"]) {
  const field = useContext(FieldContext);
  return {
    "aria-describedby": [describedBy, field?.describedBy].filter(Boolean).join(" ") || undefined,
    "aria-invalid": invalid ?? (field?.invalid ? true : undefined),
  };
}
