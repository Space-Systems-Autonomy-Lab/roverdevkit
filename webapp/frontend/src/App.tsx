import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

import { AppShell } from "@/components/app-shell";
import { DesignExplorer } from "@/pages/design-explorer";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell>
        <DesignExplorer />
      </AppShell>
      {import.meta.env.DEV ? <ReactQueryDevtools position="bottom" /> : null}
    </QueryClientProvider>
  );
}
