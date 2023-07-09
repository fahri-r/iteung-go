package main

import (
	"context"
	"encoding/gob"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/datasetter/char"

	."github.com/fahri-r/iteung-go/vocab"

	"github.com/adrg/strutil"
	"github.com/adrg/strutil/metrics"
)

type configuration struct {
	Dump string `envconfig:"dump" default:"checkpoint.bin"`
}

type backup struct {
	Model      lstm.Model
	Vocabulary Vocabulary[string, int]
}

func main() {
	
	// efore we brt we bus repetition. the superfluity say, he catunt thones not urfeits, er abe can bust ne

	var config configuration
	err := envconfig.Process("TRAIN", &config)
	if err != nil {
		log.Fatal(err)
	}

	var recovered backup

	f, err := os.Open(config.Dump)
	if err != nil {
		log.Println(err)
	}

	dec := gob.NewDecoder(f)
	err = dec.Decode(&recovered)
	if err != nil {
		log.Println(err)
	}

	model := recovered.Model
	vocab := recovered.Vocabulary

	
    data, err := ioutil.ReadFile("dataset/output/test_qa.txt")
    if err != nil {
        return
    }

    resultFile, err := os.Create("dataset/output/result_test_qa.txt")
    if err != nil {
        log.Fatal(err)
    }
    defer resultFile.Close()

	
    questionAnswerRecords := strings.Split(string(data), "\n\n")
	totalAccuracyInFloat := 0.0
	for i := 0; i < len(questionAnswerRecords); i++ {
		question := strings.Split(questionAnswerRecords[i], "\n")[0]
		
		if(strings.TrimSpace(question) == "") {
			continue
		}

		fmt.Println("Prompt:", question)


		// fmt.Printf("Vocabulary: %v\n", vocab.Size())

		parts := strings.Fields(question)
		for _, r := range parts {
			_, err := vocab.TokenToIdx(r)

			if err != nil {
			panic("Please use a prompt with known vocabulary characters")
			}
		}

		vocabSize := vocab.Size()

		prediction := char.NewPrediction(question, vocab.TokenToIdx, 100, vocabSize)

		err = model.Predict(context.TODO(), prediction)
		if err != nil {
			log.Println(err)
		}

		// fmt.Println("Prediction Size:", len(prediction.GetOutput()))

		answer:=""
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
			// fmt.Printf("%v\n", output)
			answer += strings.TrimSpace(string(rne)) + " "
		}
		fmt.Println(strings.TrimSpace(answer))

		accuracy := strutil.Similarity(questionAnswerRecords[1], strings.TrimSpace(answer), metrics.NewJaroWinkler())
		totalAccuracyInFloat += accuracy
		
		_, err := resultFile.WriteString(fmt.Sprintf("%s\n%s %f\n\n", strings.TrimSpace(question), strings.TrimSpace(answer), accuracy))

		if err != nil {
			log.Fatal(err)
		}

	}

	_, err = resultFile.WriteString(fmt.Sprintf("Accuracy: %f%%", (totalAccuracyInFloat / float64(len(questionAnswerRecords))) * 100))

		if err != nil {
			log.Fatal(err)
		}
}
