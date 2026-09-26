/* The SPA root: a data router (createBrowserRouter) over the manifest-built
 * route table, with the shell as its root layout. main.tsx provides the
 * QueryClient around it; the shell mounts UiProvider inside the router. */
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { ROUTER_FUTURE, buildRoutes } from "./routes";
import { Shell } from "./Shell";

export const router = createBrowserRouter(buildRoutes({ layout: Shell }), { future: ROUTER_FUTURE });

export default function App() {
  return <RouterProvider router={router} future={{ v7_startTransition: true }} />;
}
