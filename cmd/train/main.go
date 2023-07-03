package main

import (
	"bufio"
	"context"
	"encoding/gob"
	"fmt"
	"io"
	_"io/ioutil"
	"flag"
	"log"
	"os"

	"strings"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/datasetter/char"
	G "gorgonia.org/gorgonia"

	."github.com/fahri-r/iteung-go/vocab"
)

type configuration struct {
	Dump string `envconfig:"dump" default:"checkpoint.bin"`
}

func newVocabulary(filename string) (*Vocabulary[string, int], error) {

	v := NewVocabStructure[string, int]()

	f, err := os.Open(filename)
	if err != nil {
	    return v, err
	}
	defer f.Close()

	r := bufio.NewReader(f)

	end := false

	// special workaround to add newline token
	id := 0
	i := 0
	v.Insert("\n", id)
	i++
	id++

	// add only unique tokens to vocabulary
	for i=i; !end; i++ {
	    l, err := r.ReadString('\n')
	    if err != nil {
		end = true
	    }

	    if l == "\n" {
		continue
	    }

	    l = strings.TrimRight(l, "\n")

	    parts := strings.Fields(l)

	    for ii, p := range parts {

		_, exists := v.Get(p)
		if !exists {
		    v.Insert(p, id)
		    fmt.Println("Adding", p, "to vocabulary")
		    id++
		}

		if len(parts) != ii+1 {
		    i++
		}
	    }
	}

	return v, nil

}

type backup struct {
	Model      lstm.Model
	Vocabulary InferenceVocabulary[string, int]
}

func main() {
	
	var config configuration
	err := envconfig.Process("TRAIN", &config)
	if err != nil {
		log.Fatal(err)
	}

    filename := flag.String("i", "train_qa.txt", "input file name")
	flag.Parse()

	// Read the file
	
	vocab, err := newVocabulary("dataset/output/" + *filename)
	if err != nil {
		log.Fatal(err)
	}

	_, err = vocab.TokenToIdx("\n")
	if err != nil {
	    panic(err)
	}

	fmt.Printf("Vocabulary: %v\n", vocab.Size())

	// os.Exit(0)

	// TRAINING ARGUMENTS
	prompt := "siang"
	iter := 10

	vocabSize := vocab.Size()
	model := lstm.NewModel(vocabSize, vocabSize, 100)

	learnrate := 1e-3
	l2reg := 1e-6
	clipVal := float64(5)
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	for i := 0; i < iter; i++ {
		f, err := os.Open("dataset/output/" + *filename)
		if err != nil {
			log.Fatal(err)
		}
		max, _ := f.Seek(0, io.SeekEnd)
		
		f.Seek(0, io.SeekStart)
		
		fmt.Println("Preparing dataset...")

		tset := char.NewTrainingSet(f, vocab.TokenToIdx, vocab.IdxToToken, vocabSize, 30, 1)
		pause := make(chan struct{})
		infoChan, errc := model.Train(context.TODO(), tset, solver, pause)
		iter := 1
		var minLoss float32
		
		fmt.Printf("Starting training (%v)...\n", i)

		for infos := range infoChan {
			if iter%100 == 0 {
				if minLoss == 0 {
					minLoss = infos.Cost
				}
				if infos.Cost < minLoss {
					minLoss = infos.Cost
					log.Println("Backup because loss is minimum")

					infVocab := NewInferenceVocabFromExsting(*vocab)

					bkp := backup{
						Model:      *model,
						Vocabulary: *infVocab,
					}
					f, err := os.OpenFile(config.Dump, os.O_RDWR|os.O_CREATE, 0755)
					if err != nil {
						log.Println(err)
					}
					enc := gob.NewEncoder(f)
					err = enc.Encode(bkp)
					if err != nil {
						log.Println(err)
					}
					if err := f.Close(); err != nil {
						log.Println(err)
					}
				}
				here, _ := f.Seek(0, io.SeekCurrent)
				
				fmt.Printf("[%v/%v]%v\n", here, max, infos)

				// if here >= max {
				//     pause <- 1
				// }

			}
			if iter%500 == 0 {
				fmt.Println("\nGoing to predict")
				pause <- struct{}{}
				prediction := char.NewPrediction(prompt, vocab.TokenToIdx, 100, vocabSize)
				err := model.Predict(context.TODO(), prediction)
				if err != nil {
					log.Println(err)
					continue
				}

				for _, output := range prediction.GetOutput() {
					var idx int
					for i, val := range output {
						if val == 1 {
							idx = i
						}
					}
					rne, err := vocab.IdxToToken(idx)
					if err != nil {
						log.Fatal(err)
					}
					fmt.Printf(rne)
				}
				fmt.Println("")
				pause <- struct{}{}
			}
			iter++
		}
		err = <-errc
		if err == io.EOF {
			close(pause)
			//return
		}
		if err != nil && err != io.EOF {
			log.Fatal(err)
		}
		f.Close()
	}

	fmt.Println("Done")

}