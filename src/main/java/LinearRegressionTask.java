import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;

public class LinearRegressionTask {

    private static final int ITEMS_NUM = 100;

    private static final String DEPENDENT_VALUES = "Chance of Admit";

    private static final String INDEPENDENT_VALUES = "GRE Score";

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LinearRegression")
                .master("local[*]")
                .getOrCreate();

        // Завантаження даних з CSV файлу у DataFrame
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .csv("adm_data.csv")
                .limit(ITEMS_NUM);

        // Вибірка потрібних стовпців
        Dataset<Row> selectedData =
                data.select(col(INDEPENDENT_VALUES).cast("int"), col(DEPENDENT_VALUES).cast("double"));

        // Побудова моделі лінійної регресії
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{INDEPENDENT_VALUES})
                .setOutputCol("features");

        Dataset<Row> inputData = assembler.transform(selectedData);

        LinearRegression lr = new LinearRegression()
                .setFeaturesCol("features")
                .setLabelCol(DEPENDENT_VALUES);

        LinearRegressionModel lrModel = lr.fit(inputData);

        // Виведення коефіцієнтів регресії
        double slope = lrModel.coefficients().apply(0);
        double intercept = lrModel.intercept();
        System.out.println("Нахил: " + slope);
        System.out.println("Зсув: " + intercept);

        // Оцінка моделі
        Dataset<Row> predictions = lrModel.transform(inputData);
        predictions.show();

        System.out.println("Коефіцієнт детермінації (R2): " + lrModel.summary().r2());

        double correlation = selectedData.stat().corr(INDEPENDENT_VALUES, DEPENDENT_VALUES);
        System.out.println("Кореляція між балом GRE і шансом прийняття: " + correlation);

        System.out.println("\nP-значення: " + Arrays.toString(lrModel.summary().pValues()));

        double meanPrediction = predictions.select("prediction").agg(avg("prediction")).head().getDouble(0);
        System.out.println("\nПрогнозоване середнє значення залежної змінної: " + meanPrediction);

        // Середня квадратична помилка
        double stdErr = lrModel.summary().rootMeanSquaredError();

        List<Double> greScores = selectedData.select(INDEPENDENT_VALUES).as(Encoders.DOUBLE()).collectAsList();

        // Довірчий інтервал коефіцієнта регресії
        double xAvg = greScores.stream()
                .mapToInt(Double::intValue)
                .average().orElse(0);

        double xDeviationSquaredSum = greScores.stream()
                .mapToDouble(x -> Math.pow(x - xAvg, 2))
                .sum();

        double tValue = 1.9845; // для 95% надійності з 98 ступенями вільності
        double lowerBound = slope - tValue * (stdErr / Math.sqrt(xDeviationSquaredSum));
        double upperBound = slope + tValue * (stdErr / Math.sqrt(xDeviationSquaredSum));
        System.out.println("Довірчий інтервал: [" + lowerBound + ", " + upperBound + "]");

        // Довірчий інтервал для кожного X
        greScores.sort(Comparator.naturalOrder());
        List<Double> lowerBoundsForEachScore = greScores.stream().map(x -> (intercept + slope * x) -
                        tValue * (stdErr * Math.sqrt((double) 1 / ITEMS_NUM + (Math.pow(x - xAvg, 2) / xDeviationSquaredSum))))
                .toList();

        List<Double> upperBoundsForEachScore = greScores.stream().map(x -> (intercept + slope * x) +
                        tValue * (stdErr * Math.sqrt((double) 1 / ITEMS_NUM + (Math.pow(x - xAvg, 2) / xDeviationSquaredSum))))
                .toList();

        List<Double> scores = selectedData.select(INDEPENDENT_VALUES).as(Encoders.DOUBLE()).collectAsList();
        List<Double> chances = selectedData.select(DEPENDENT_VALUES).as(Encoders.DOUBLE()).collectAsList();
        new ChartCreator().create("Залежність шансу вступу від балу GRE", "Бал GRE", "Шанс вступу")
                .addScatterChart("Дані", scores, chances)
                .addLinearRegressionChart(String.format("y = %.3f * x + %.3f", slope, intercept), greScores.get(0),
                        greScores.get(greScores.size() - 1), slope, intercept)
                .addSeries("Нижня межа довірчого інтервалу", greScores, lowerBoundsForEachScore)
                .addSeries("Верхня межа довірчого інтервалу", greScores, upperBoundsForEachScore)
                .build("chart");

        spark.stop();
    }
}
