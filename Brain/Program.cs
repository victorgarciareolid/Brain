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
            for(int i=0; i<10; i++)
            {
                Test();
            }
            Console.ReadKey();
        }

        static void printAr(double[] array)
        {
            Console.Write("[");
            for(int i=0; i<array.Length-1; i++)
            {
                Console.Write(array[i].ToString() + ", ");
            }
            Console.WriteLine(array[array.Length - 1] + "]");
        }

        static void Test()
        {
            Random ran = new Random();
            double[] input = { 0.0, 0.0 };
            double[] output;
            double[] aux_i = new double[2];
            double[] aux_o = new double[1];
            int N = 1000;
            int i1, i2;
            TrainPair[] tps = new TrainPair[N];

            for (int i = 0; i < N; i++)
            {
                i1 = 0;
                i2 = 0;
                if (ran.NextDouble() > 0.5) i1 = 1;
                if (ran.NextDouble() > 0.5) i2 = 1;

                aux_i[0] = i1;
                aux_i[1] = i2;
                aux_o[0] = (double)(i1 ^ i2);

                //Console.WriteLine("{0} ^ {1} = {2}", aux_i[0], aux_i[1], aux_o[0]);

                tps[i] = new TrainPair(aux_i, aux_o);
            }


            TrainSet ts = new TrainSet(tps);

            Red r = new Red(2, 2, 1);

            //printAr(output);
            r.train(ts, 1);
            output = r.compute(input);
            Console.WriteLine("0 ^ 0 = " + Math.Round(output[0]) + " ({0})", output[0]);

            input[0] = 1.0;
            input[1] = 0.0;
            output = r.compute(input);
            Console.WriteLine("0 ^ 1 = " + Math.Round(output[0]) + " ({0})", output[0]);

            input[0] = 0.0;
            input[1] = 1.0;
            output = r.compute(input);
            Console.WriteLine("1 ^ 0 = " + Math.Round(output[0]) + " ({0})", output[0]);

            input[0] = 1.0;
            input[1] = 1.0;
            output = r.compute(input);
            Console.WriteLine("1 ^ 1 = " + Math.Round(output[0]) + " ({0})", output[0]);

            Console.WriteLine("*******************");
        }
    }
}
