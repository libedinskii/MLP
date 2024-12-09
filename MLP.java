import java.util.Random;

public class MLP {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final double[][] hiddenWeights; // Вес для скрытого слоя
    private final double[] hiddenBias;       // Сдвиг для скрытого слоя
    private final double[][] outputWeights;  // Вес для выходного слоя
    private final double[] outputBias;       // Сдвиг для выходного слоя
    private final Random random;

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.hiddenWeights = new double[inputSize][hiddenSize];
        this.outputWeights = new double[hiddenSize][outputSize];
        this.hiddenBias = new double[hiddenSize];
        this.outputBias = new double[outputSize];
        this.random = new Random();

        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                hiddenWeights[i][j] = random.nextDouble() - 0.5; // Инициализация весов
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                outputWeights[i][j] = random.nextDouble() - 0.5; // Инициализация весов
            }
            hiddenBias[i] = random.nextDouble() - 0.5; // Инициализация сдвига
        }
        for (int i = 0; i < outputSize; i++) {
            outputBias[i] = random.nextDouble() - 0.5; // Инициализация сдвига
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public double forward(double[] inputs) {
        double[] hiddenOutputs = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            hiddenOutputs[j] = 0;
            for (int i = 0; i < inputSize; i++) {
                hiddenOutputs[j] += inputs[i] * hiddenWeights[i][j];
            }
            hiddenOutputs[j] += hiddenBias[j];
            hiddenOutputs[j] = sigmoid(hiddenOutputs[j]);
        }

        double[] outputOutputs = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            outputOutputs[j] = 0;
            for (int i = 0; i < hiddenSize; i++) {
                outputOutputs[j] += hiddenOutputs[i] * outputWeights[i][j];
            }
            outputOutputs[j] += outputBias[j];
            outputOutputs[j] = sigmoid(outputOutputs[j]);
        }

        return outputOutputs[0]; // для XOR выход один
    }

    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int sample = 0; sample < trainingInputs.length; sample++) {
                // Forward pass
                double[] inputs = trainingInputs[sample];
                double[] output = trainingOutputs[sample];

                // Прямой проход
                double[] hiddenOutputs = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    hiddenOutputs[j] = 0;
                    for (int i = 0; i < inputSize; i++) {
                        hiddenOutputs[j] += inputs[i] * hiddenWeights[i][j];
                    }
                    hiddenOutputs[j] += hiddenBias[j];
                    hiddenOutputs[j] = sigmoid(hiddenOutputs[j]);
                }

                double[] finalOutputs = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    finalOutputs[j] = 0;
                    for (int i = 0; i < hiddenSize; i++) {
                        finalOutputs[j] += hiddenOutputs[i] * outputWeights[i][j];
                    }
                    finalOutputs[j] += outputBias[j];
                    finalOutputs[j] = sigmoid(finalOutputs[j]);
                }

                // Обратное распространение
                double[] outputErrors = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    outputErrors[j] = output[sample][j] - finalOutputs[j];
                }

                // Корректировка весов выходного слоя
                for (int j = 0; j < outputSize; j++) {
                    for (int i = 0; i < hiddenSize; i++) {
                        outputWeights[i][j] += learningRate * outputErrors[j] * sigmoidDerivative(finalOutputs[j]) * hiddenOutputs[i];
                    }
                    outputBias[j] += learningRate * outputErrors[j] * sigmoidDerivative(finalOutputs[j]);
                }

                // Корректировка весов скрытого слоя
                double[] hiddenErrors = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    hiddenErrors[j] = 0;
                    for (int k = 0; k < outputSize; k++) {
                        hiddenErrors[j] += outputErrors[k] * outputWeights[j][k];
                    }
                }

                for (int j = 0; j < hiddenSize; j++) {
                    for (int i = 0; i < inputSize; i++) {
                        hiddenWeights[i][j] += learningRate * hiddenErrors[j] * sigmoidDerivative(hiddenOutputs[j]) * inputs[i];
                    }
                    hiddenBias[j] += learningRate * hiddenErrors[j] * sigmoidDerivative(hiddenOutputs[j]);
                }
            }
        }
    }

    public static void main(String[] args) {
        MLP mlp = new MLP(2, 2, 1); // 2 входа, 2 скрытых нейрона, 1 выход

        // Данные для обучения (XOR)
        double[][] trainingInputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        };

        double[][] trainingOutputs = {
                {0}, // 0 XOR 0 = 0
                {1}, // 0 XOR 1 = 1
                {1}, // 1 XOR 0 = 1
                {0}, // 1 XOR 1 = 0
        };

        // Обучение сети
        mlp.train(trainingInputs, trainingOutputs, 10000, 0.1);

        // Тестирование
        for (double[] input : trainingInputs) {
            double output = mlp.forward(input);
            System.out.println("Input: " + input[0] + ", " + input[1] + " => Output: " + Math.round(output));
        }
    }
}
