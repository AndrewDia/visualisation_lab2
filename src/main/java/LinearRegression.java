import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
public class LinearRegression {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LinearRegression")
                .master("local[*]")
                .getOrCreate();

        // Завантаження даних з CSV файлу у DataFrame
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .csv("adm_data.csv")
                .limit(100);

        // Вибірка потрібних стовпців
        Dataset<Row> selectedData =
                data.select(col("GRE Score").cast("int"), col("Chance of Admit").cast("double"));

        // Побудова моделі лінійної регресії
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"GRE Score"})
                .setOutputCol("features");

        Dataset<Row> inputData = assembler.transform(selectedData);

        org.apache.spark.ml.regression.LinearRegression lr = new org.apache.spark.ml.regression.LinearRegression()
                .setFeaturesCol("features")
                .setLabelCol("Chance of Admit");

        LinearRegressionModel lrModel = lr.fit(inputData);

        // Виведення коефіцієнтів регресії
        System.out.println("Коефіцієнт регресії: " + lrModel.coefficients());
        System.out.println("Зсув: " + lrModel.intercept());

        // Оцінка моделі
        Dataset<Row> predictions = lrModel.transform(inputData);
        predictions.show();

        System.out.println("Коефіцієнт детермінації (R2): " + lrModel.summary().r2());

        double correlation = selectedData.stat().corr("GRE Score", "Chance of Admit");
        System.out.println("Кореляція між балом GRE і шансом прийняття: " + correlation);

        System.out.println("\nP-значення: " + Arrays.toString(lrModel.summary().pValues()));

        double meanPrediction = predictions.select("prediction").agg(avg("prediction")).head().getDouble(0);
        System.out.println("\nСереднє значення залежної змінної: " + meanPrediction);

        // Стандартна похибка коефіцієнта
        DenseVector coefficients = (DenseVector) lrModel.coefficients();
        double stdErr = lrModel.summary().coefficientStandardErrors()[0];

        double tValue = 1.9845; // для 95% надійності з 98 ступенями вільності
        double lowerBound = coefficients.apply(0) - tValue * stdErr;
        double upperBound = coefficients.apply(0) + tValue * stdErr;
        System.out.println("Довірчий інтервал: [" + lowerBound + ", " + upperBound + "]");
        spark.stop();
    }
}
