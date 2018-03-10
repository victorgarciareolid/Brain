using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Brain.NN
{
    class TrainPair
    {
        private double[] inputs;
        private double[] outputs;

        public TrainPair(double[] entrada, double[] salida)
        {
            inputs = entrada;
            outputs = salida;
        }

        public double[] getInputs()
        {
            return inputs;
        }

        public double[] getOutputs()
        {
            return outputs;
        }
    }
}
