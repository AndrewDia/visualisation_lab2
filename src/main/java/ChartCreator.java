import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.IOException;
import java.util.List;

public class ChartCreator {

    private XYChart chart;

    public ChartCreator create(String title, String xAxisTitle, String yAxisTitle) {
        chart = new XYChartBuilder()
                .width(800).height(600)
                .title(title)
                .xAxisTitle(xAxisTitle).yAxisTitle(yAxisTitle)
                .build();
        return this;
    }

    public ChartCreator addScatterChart(String seriesName, List<Double> xData, List<Double> yData) {
        chart.addSeries(seriesName, xData, yData).setLineStyle(SeriesLines.NONE);
        return this;
    }

    public ChartCreator addLinearRegressionChart(String seriesName, double from, double to, double slope, double intercept) {
        double[] xBoundaries = {from, to};
        double[] yBoundaries = {slope * from + intercept, slope * to + intercept};
        chart.addSeries(seriesName, xBoundaries, yBoundaries).setMarker(SeriesMarkers.NONE);
        return this;
    }

    public ChartCreator addSeries(String seriesName, List<Double> xData, List<Double> yData) {
        chart.addSeries(seriesName, xData, yData).setMarker(SeriesMarkers.NONE);
        return this;
    }

    public void build(String fileName) {
        try {
            BitmapEncoder.saveBitmap(chart, fileName + ".png", BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}