import { Play } from "lucide-react";

import { DesignForm } from "@/components/design-form";
import { PredictionPanel } from "@/components/prediction-panel";
import { ScenarioPicker } from "@/components/scenario-picker";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { usePredict } from "@/hooks/use-predict";
import { useDesignStore } from "@/store/design-store";

/**
 * Single-design panel: scenario picker + 12-D design form on the
 * left, prediction (median + 90 % PI) on the right.
 */
export function DesignExplorer() {
  const design = useDesignStore((s) => s.design);
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const predict = usePredict();

  const handlePredict = () => {
    predict.mutate({ design, scenario_name: scenarioName });
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Card>
        <CardHeader>
          <CardTitle>Design vector</CardTitle>
          <CardDescription>
            12-D rover design plus a canonical mission scenario. Bounds match
            the v3-widened LHS sampling; the backend will reject anything
            outside them with a 422.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <ScenarioPicker />
          <DesignForm disabled={predict.isPending} />
          <Button
            type="button"
            onClick={handlePredict}
            disabled={predict.isPending}
            className="w-full"
            size="lg"
          >
            <Play className="mr-2 h-4 w-4" />
            {predict.isPending ? "Evaluating…" : "Run surrogate"}
          </Button>
        </CardContent>
      </Card>

      <PredictionPanel
        data={predict.data}
        isPending={predict.isPending}
        error={predict.error}
      />
    </div>
  );
}
