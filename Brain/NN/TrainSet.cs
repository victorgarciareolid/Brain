using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Brain.NN
{
    class TrainSet
    {
        List<TrainPair> data;

        public TrainSet(TrainPair[] input)
        {
            data = new List<TrainPair>();
            foreach(TrainPair tp in input)
            {
                data.Add(tp);
            }
        }

        public TrainPair getPair()
        {
            TrainPair aux = data[data.Count - 1];
            data.Remove(aux);
            return aux;
        }
        
        public bool finish()
        {
            return data.Count == 0;
        }
    }
}
