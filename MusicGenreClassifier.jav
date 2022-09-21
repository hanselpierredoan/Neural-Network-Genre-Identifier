import org.tensorflow.*;
import java.util.ArrayList;
import java.util.List;
import java.io.*;

public class MusicGenreClassifier {
    // Loads the dataset
    public static List<String[]> loadDataset(String fileName) throws IOException {
        List<String[]> dataset = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] fields = line.split("\t");
            dataset.add(fields);
        }
        reader.close();
        return dataset;
    }

    // Creates the neural network
    public static void main(String[] args) {
        try {
            List<String[]> dataset = loadDataset("GITZAN.txt");
            
            // Input layer (song features)
            final int NUM_FEATURES = dataset.get(0).length - 1;
            final int NUM_CLASSES = 10; // 10 genres

            // Output layer (genre prediction)
            final long seed = 0;
            final int NUM_HIDDEN = 10;
            final int NUM_ITERATIONS = 10000;
            final double LEARNING_RATE = 0.5;

            // Split into train and test sets
            int trainSize = (int) (dataset.size() * 0.8);
            int testSize = dataset.size() - trainSize;

            List<String[]> trainSet = dataset.subList(0, trainSize);
            List<String[]> testSet = dataset.subList(trainSize, dataset.size());

            // Convert to float arrays
            float[][] trainFeatures = new float[trainSize][NUM_FEATURES];
            float[][] trainLabels = new float[trainSize][NUM_CLASSES];
            for (int i = 0; i < trainSize; i++) {
                String[] example = trainSet.get(i);
                for (int j = 0; j < NUM_FEATURES; j++) {
                    trainFeatures[i][j] = Float.parseFloat(example[j]);
                }
                int label = Integer.parseInt(example[NUM_FEATURES]);
                trainLabels[i][label] = 1.0f;
            }

            float[][] testFeatures = new float[testSize][NUM_FEATURES];
            float[][] testLabels = new float[testSize][NUM_CLASSES];
            for (int i = 0; i < testSize; i++) {
                String[] example = testSet.get(i);
                for (int j = 0; j < NUM_FEATURES; j++) {
                    testFeatures[i][j] = Float.parseFloat(example[j]);
                }
                int label = Integer.parseInt(example[NUM_FEATURES]);
                testLabels[i][label] = 1.0f;
            }

            // Create the neural network
            try (TensorFlow tf = TensorFlow.create()) {
                // Define the layer sizes
                int numInputs = NUM_FEATURES;
                int numHidden = NUM_HIDDEN;
                int numOutputs = NUM_CLASSES;

                // Create variables for the hidden and output layers
                Variable hiddenWeights = tf.variable("hiddenWeights", tf.zeros(numInputs, numHidden));
                Variable hiddenBiases = tf.variable("hiddenBiases", tf.zeros(numHidden));
                Variable outputWeights = tf.variable("outputWeights", tf.zeros(numHidden, numOutputs));
                Variable outputBiases = tf.variable("outputBiases", tf.zeros(numOutputs));

                // Create placeholders for the inputs and outputs
                Placeholder<Float> input = tf.placeholder(Float.class, Shape.of(numInputs));
                Placeholder<Float> labels = tf.placeholder(Float.class, Shape.of(numOutputs));

                // Define the hidden layer
                Operation hidden = tf.nn.relu(
                        tf.matmul(input, hiddenWeights).add(hiddenBiases)
                );

                // Define the output layer
                Operation output = tf.nn.softmax(
                        tf.matmul(hidden, outputWeights).add(outputBiases)
                );

                // Define the loss function
                Operation loss = tf.reduce_mean(
                        tf.square(output.substract(labels))
                );

                // Define the training step
                Operation trainStep = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss);

                // Initialize the variables
                try (Session session = tf.session()) {
                    tf.globalVariablesInitializer().run(session);

                    // Train the network
                    for (int i = 0; i < NUM_ITERATIONS; i++) {
                        float[][] batchFeatures = new float[trainSize][numInputs];
                        float[][] batchLabels = new float[trainSize][numOutputs];
                        for (int j = 0; j < trainSize; j++) {
                            int index = (int) (Math.random() * trainSize);
                            batchFeatures[j] = trainFeatures[index];
                            batchLabels[j] = trainLabels[index];
                        }

                        session.run(
                                trainStep,
                                tf.feed(input, batchFeatures),
                                tf.feed(labels, batchLabels)
                        );
                    }

                    // Evaluate the model
                    float accuracy = tf.evaluate(
                            output,
                            testFeatures,
                            testLabels
                    );
                    System.out.println("Accuracy: " + accuracy);

                    // Save the model
                    tf.savedModel("model/", input, output);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
