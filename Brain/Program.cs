using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Brain.NN;

namespace Brain
{
    class Program
    {
        static void Main(string[] args)
        {
            Random ran = new Random();
            double[] input = { 1.0, 0.0 };
            double[] aux_i = new double[2];
            double[] aux_o = new double[1];
            
            int N = 100;
            int i1, i2;
            TrainPair[] tps = new TrainPair[N];

            for(int i=0; i<N; i++)
            {
                i1 = 0;
                i2 = 0;
                if (ran.NextDouble() > 0.5) i1 = 1;
                if (ran.NextDouble() > 0.5) i2 = 1;

                aux_i[0] = i1;
                aux_i[1] = i2;
                aux_o[0] = (double)(i1 ^ i2);

                tps[i] = new TrainPair(aux_i, aux_o);
            }


            TrainSet ts = new TrainSet(tps);

            Red r = new Red(2, 2, 1);

            double[] output = r.compute(input);
            printAr(output);

            r.train(ts, 0.6);
            printAr(output);

            Console.WriteLine(Perceptron.sigmoid(0.7));

            Console.ReadKey();
        }

        static void printAr(double[] array)
        {
            for(int i=0; i<array.Length-1; i++)
            {
                Console.Write(array[i].ToString() + ", ");
            }
            Console.WriteLine(array[array.Length - 1]);
        }
    }
}
