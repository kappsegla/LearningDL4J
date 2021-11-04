package demo;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY;

public class MinstClassifier {

    private static final String RESOURCES_FOLDER_PATH = "./src/main/resources/mnist_png";

    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;

    private static final int N_SAMPLES_TRAINING = 60000;
    private static final int N_SAMPLES_TESTING = 10000;

    private static final int N_OUTCOMES = 10;
    private static long t0 = System.currentTimeMillis();

    public static void main(String[] args) throws IOException {

        t0 = System.currentTimeMillis();
        //System.out.print(RESOURCES_FOLDER_PATH + "/training");
        DataSetIterator dataSetIterator = getDataSetIterator(RESOURCES_FOLDER_PATH + "/training", N_SAMPLES_TRAINING);

        buildModel(dataSetIterator);
    }

    private static void buildModel(DataSetIterator dsi) throws IOException {

        int rngSeed = 123;
        int nEpochs = 1;

        System.out.println("Build Model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4).list()
                .layer(new DenseLayer.Builder()
                        .nIn(HEIGHT * WIDTH).nOut(1000).activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000).nOut(N_OUTCOMES).activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //Print score every 500 interaction
        model.setListeners(new ScoreIterationListener(500));

        System.out.println("Train Model...");
        model.fit(dsi, nEpochs);

        //Evaluation
        DataSetIterator testDsi = getDataSetIterator(RESOURCES_FOLDER_PATH + "/testing", N_SAMPLES_TESTING);
        System.out.print("Evaluating Model...");
        Evaluation eval = model.evaluate(testDsi);
        System.out.print(eval.stats());

        long t1 = System.currentTimeMillis();
        double t = (double) (t1 - t0) / 1000.0;
        System.out.println("\n\nTotal time: " + t + " seconds");

        //Use
        System.out.println("Classify images...");
        DataSet allData = getInputData(RESOURCES_FOLDER_PATH + "/myimages");

        INDArray output = model.output(allData.getFeatures());

        for(int i = 0; i < output.rows(); i++)
        {
            System.out.println("Image " + i + " classified as:");
            for(int j = 0; j < 10; j++)
                System.out.printf("%d : %.2f  ",j, output.getRow(i).getDouble(j));
            System.out.println("");
        }
    }

    private static DataSet getInputData(String folderPath) throws IOException {
        try {
            File folder = new File(folderPath);

            NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH,1, new ColorConversionTransform(COLOR_RGBA2GRAY)); //28x28
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1); //translate image into seq of 0..1 input values

            File[] imageFiles = folder.listFiles();
            int imageCount = imageFiles.length;

            INDArray input = Nd4j.create(imageCount, HEIGHT * WIDTH);
            INDArray output = Nd4j.create(imageCount, N_OUTCOMES);

            int n = 0;

            for (File imgFile : imageFiles) {
                INDArray img = nativeImageLoader.asRowVector(imgFile);
                scaler.transform(img);
                input.putRow(n, img);
                n++;
            }
            //Joining input and output matrices into a dataset
            return new DataSet(input, output);
        } catch (
                Exception e) {
            System.out.println(e.getLocalizedMessage());
            return null;
        }
    }

    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
        try {
            File folder = new File(folderPath);
            File[] digitFolders = folder.listFiles();

            NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH); //28x28
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1); //translate image into seq of 0..1 input values

            INDArray input = Nd4j.create(nSamples, HEIGHT * WIDTH);
            INDArray output = Nd4j.create(nSamples, N_OUTCOMES);

            int n = 0;
            //scan all 0 to 9 digit subfolders
            for (File digitFolder : digitFolders) {
                int labelDigit = Integer.parseInt(digitFolder.getName());
                File[] imageFiles = digitFolder.listFiles();

                for (File imgFile : imageFiles) {
                    INDArray img = nativeImageLoader.asRowVector(imgFile);
                    scaler.transform(img);
                    input.putRow(n, img);
                    output.put(n, labelDigit, 1.0);
                    n++;
                }
            }

            //Joining input and output matrices into a dataset
            DataSet dataSet = new DataSet(input, output);
            //Convert the dataset into a list
            List<DataSet> listDataSet = dataSet.asList();
            //Shuffle content of list randomly
            Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
            int batchSize = 10;

            //Build and return a dataset iterator
            return new ListDataSetIterator<DataSet>(listDataSet, batchSize);
        } catch (Exception e) {
            System.out.println(e.getLocalizedMessage());
            return null;
        }
    }
}