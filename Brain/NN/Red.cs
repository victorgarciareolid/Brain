using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Brain.NN
{
    class Red
    {
        private double[] input;
        private Perceptron[] interior;
        private Perceptron[] output;
        private int n_input;
        private int n_interior;
        private int n_output;

        public Red(int n_input, int n_interior, int n_output)
        {
            this.n_input = n_input;
            this.n_interior = n_interior;
            this.n_output = n_output;
            input = new double[n_input];
            interior = new Perceptron[n_interior];
            output = new Perceptron[n_output];

            for (int i = 0; i < n_interior; i++)
            {
                interior[i] = new Perceptron(n_input, 0);
            }
            for (int i = 0; i < n_output; i++)
            {
                output[i] = new Perceptron(n_interior, 0);
            }
        }

        private Tuple<double[], double[]> forward(double[] entrada)
        {
            double[] computed_out = new double[n_output];
            double[] computed_interior = new double[n_interior];

            for (int i = 0; i < n_interior; i++)
            {
                computed_interior[i] = interior[i].think(entrada);
            }
            for (int i = 0; i < n_output; i++)
            {
                computed_out[i] = output[i].think(computed_interior);
            }

            return Tuple.Create<double[], double[]>(computed_interior, computed_out);
        }

        public void train(TrainSet ts, double learning_rate)
        {
            double[,] aux_partial = new double[n_output, n_interior];
            double[] salida = new double[n_output];
            Tuple<double[], double[]> aux;
            double[] hidden = new double[n_interior];
            double total_error=0, updated_weight;

            while (!ts.finish())
            {
                TrainPair tp = ts.getPair();

                aux = forward(tp.getInputs());
                hidden = aux.Item1;
                salida = aux.Item2;

                total_error = error(tp, salida);

                //Console.WriteLine("Current Network Error: {0}", total_error);

                // Train output
                for (int i = 0; i<n_output; i++)
                {
                    for (int j=0; j<n_interior; j++)
                    {
                        aux_partial[i, j] = -(tp.getOutputs()[i] - salida[i]) * derivada(salida[i]) * hidden[j];
                        updated_weight = output[i].getWeights()[j] - learning_rate * aux_partial[i, j];
                        output[i].updateWeight(j, updated_weight);
                    }
                }
                aux_partial = new double[n_interior, n_input];
                // Train hidden
                for (int i = 0; i < n_interior; i++)
                {
                    for (int j = 0; j < n_input; j++)
                    {
                        double deltasum=0;

                        for (int k=0; k<n_output; k++)
                        {
                            deltasum += output[k].getWeights()[i]* (-(tp.getOutputs()[k] - salida[k]) * derivada(salida[k]));
                        }

                        aux_partial[i, j] = deltasum * derivada(hidden[i]) * input[j];

                        updated_weight = interior[i].getWeights()[j] - learning_rate * aux_partial[i , j];

                        interior[i].updateWeight(j, updated_weight);
                    }
                }
            }
            Console.WriteLine("Net Error: " + total_error);
        }

        private double derivada(double u)
        {
            return u*(1.0-u);
        }

        private double error(TrainPair tp, double[] salida)
        {
            double aux=0;
            for(int i=0; i<n_output; i++)
            {
                aux += Math.Pow(tp.getOutputs()[i] - salida[i], 2);
            }
            return aux/2;
        }
        
        public double[] compute(double[] entrada)
        {
            return forward(entrada).Item2;
        }
        
    }
}
