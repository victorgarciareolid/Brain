using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Brain.NN
{
    class Perceptron
    {
        private double[] weights;
        private int n_inputs;
        private double bias = 0;

        public Perceptron(int n_inputs, double bias)
        {
            Random aux = new Random();
            this.n_inputs = n_inputs;
            this.bias = bias;
            weights = new double[n_inputs];

            for(int i=0; i<n_inputs; i++)
            {
                weights[i] = aux.NextDouble();
            }
        }

        public double think(double[] inputs)
        {
            double aux = 0;
            for(int i=0; i<n_inputs; i++)
            {
                aux += inputs[i] * weights[i];
            }
            return sigmoid(aux + bias);
        }

        public static double sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        public void setWeights(double[] w)
        {
            w.CopyTo(weights, 0);
        }

        public double[] getWeights()
        {
            return weights;
        }

        public void updateWeight(int index, double value)
        {
            weights[index] = value;
        }

    }
}
