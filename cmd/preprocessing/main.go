package main

import (
    "encoding/csv"
    "io"
    "log"
    "os"
    "regexp"
    "strings"
	"github.com/RadhiFadlillah/go-sastrawi"
    "github.com/go-gota/gota/dataframe"
    "fmt"
    "flag"
	"io/ioutil"
    "math/rand"
    "math"
)

var punct_re_escape *regexp.Regexp
var dataSlang = make(map[string]string)
var dictionary sastrawi.Dictionary
var stemmer sastrawi.Stemmer

type QuestionAnswerLength struct{
    DataLength int
    TotalSentence int
}

func init() {
    punct_re_escape = regexp.MustCompile("[" + regexp.QuoteMeta("!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~") + "]")
    
    dictionary = sastrawi.DefaultDictionary()
    stemmer = sastrawi.NewStemmer(dictionary)
    
    f, err := os.Open("./dataset/daftar-slang-bahasa-indonesia.csv")
    if err != nil {
        log.Fatal(err)
    }

    // remember to close the file at the end of the program
    defer f.Close()

    // read csv values using csv.Reader
    csvReader := csv.NewReader(f)
	csvReader.FieldsPerRecord = -1
    csvReader.Comma = ','
    for {
        rec, err := csvReader.Read()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatal(err)
        }

        dataSlang[rec[0]] = rec[1]
    }
}

func main() {
    input := flag.String("i", "qa.csv", "input file name")
    output := flag.String("o", "qa.txt", "output file name")
    flag.Parse()
    trainPercent := 80.0
    // testPercent := 20
    
    var questionLength = make(map[int]int)
    var answerLength = make(map[int]int)
    
    // open file
    f, err := os.Open("dataset/" + *input)
    if err != nil {
        log.Fatal(err)
    }

    // remember to close the file at the end of the program
    defer f.Close()

    // read csv values using csv.Reader
    csvReader := csv.NewReader(f)
	csvReader.FieldsPerRecord = -1
    csvReader.Comma = '|'

    for {
        rec, err := csvReader.Read()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatal(err)
        }

        question := rec[0]
        answer := ""
        if len(rec) > 1 {
            answer = rec[1]
        }

        question = NormalizeSentence(question)
        question = NormalizeSentence(question)
        question = stemmer.Stem(question)
        question = strings.TrimSpace(question)
        
        _, ok := questionLength[len(strings.Split(question, " "))]
        if ok {
            questionLength[len(strings.Split(question, " "))] += 1
        } else {
            questionLength[len(strings.Split(question, " "))] = 1
        }

        answer = strings.TrimSpace(answer)
        _, ok = answerLength[len(strings.Split(answer, " "))]
        if ok {
            answerLength[len(strings.Split(answer, " "))] += 1
        } else {
            answerLength[len(strings.Split(answer, " "))] = 1
        }
    }

    var questionLengthArr []QuestionAnswerLength
    for  key, value := range questionLength {
        data := QuestionAnswerLength{key, value}
        questionLengthArr= append(questionLengthArr, data)
    }

    dfQuestionLength := dataframe.LoadStructs(questionLengthArr)
    dfQuestionLength = dfQuestionLength.Arrange(
        dataframe.Sort("DataLength"),
    )

    var answerLengthArr []QuestionAnswerLength
    for  key, value := range answerLength {
        data := QuestionAnswerLength{key, value}
        answerLengthArr= append(answerLengthArr, data)
    }

    dfAnswerLength := dataframe.LoadStructs(answerLengthArr)
    dfAnswerLength = dfAnswerLength.Arrange(
        dataframe.Sort("DataLength"),
    )

 
    f, err = os.Open("dataset/" + *input)
    reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1
    reader.Comma = '|'
    records, _ := reader.ReadAll()

    recordsLength := len(records)
    shuffledRecords := rand.Perm(recordsLength)

    f, err = os.Create("dataset/output/" + *output)
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    for i, _ := range records {
        index:=shuffledRecords[i]
        
        if i == 0 {
            continue
        }

        question := records[index][0]
        answer := ""
        if len(records[index]) > 1 {
            answer = records[index][1]
        }

        question = NormalizeSentence(question)
        question = NormalizeSentence(question)
        question = stemmer.Stem(question)

        answer = strings.ToLower(answer)
        answer = strings.Replace(answer, "iteung", "aku", -1)
        answer = strings.Replace(answer, "\n", " ", -1)
        
        if len(strings.Split(question, " ")) > 0 && len(strings.Split(question, " ")) < 13 && len(strings.Split(answer, " ")) < 29{
            if (i == (recordsLength - 1)) {
                _, err := f.WriteString(fmt.Sprintf("%s\n%s", strings.TrimSpace(question), answer))

                if err != nil {
                    log.Fatal(err)
                }
            } else {
                _, err := f.WriteString(fmt.Sprintf("%s\n%s\n\n", strings.TrimSpace(question), answer))

                if err != nil {
                    log.Fatal(err)
                }
            }
        }
    }
	
    qa, err := ioutil.ReadFile("dataset/output/qa.txt")
    if err != nil {
        return
    }

    trainFile, err := os.Create("dataset/output/train_" + *output)
    if err != nil {
        log.Fatal(err)
    }
    defer trainFile.Close()


    testFile, err := os.Create("dataset/output/test_" + *output)
    if err != nil {
        log.Fatal(err)
    }
    defer testFile.Close()

    questionAnswerRecords := strings.Split(string(qa), "\n\n")
    trainDataLength := int(math.Round(float64(len(questionAnswerRecords))*trainPercent/100))
    testDataLength := len(questionAnswerRecords) - trainDataLength

    // Splitting Data
    for i, qa := range questionAnswerRecords {
            //if training index
            if i < int(math.Round(float64(len(questionAnswerRecords))*trainPercent/100)) {
                if (i == (trainDataLength - 1)) {
                    _, err := trainFile.WriteString(fmt.Sprintf("%s", qa))

                    if err != nil {
                        log.Fatal(err)
                    }
                } else {
                    _, err := trainFile.WriteString(fmt.Sprintf("%s\n\n", qa))

                    if err != nil {
                        log.Fatal(err)
                    }
                }
            } else {
                if (i == (len(questionAnswerRecords) - 1)) {
                    _, err := testFile.WriteString(fmt.Sprintf("%s", qa))

                    if err != nil {
                        log.Fatal(err)
                    }
                } else {
                    _, err := testFile.WriteString(fmt.Sprintf("%s\n\n", qa))

                    if err != nil {
                        log.Fatal(err)
                    }
                }
            }
    }

    fmt.Println("Record Length: ", len(questionAnswerRecords))
    fmt.Println("Train Data Length: ", trainDataLength)
    fmt.Println("Test Data Length: ", testDataLength)
}

func NormalizeSentence(sentence string) string {
    sentence = punct_re_escape.ReplaceAllString(strings.ToLower(sentence), "")
    replacableWords := [8]string{"iteung", "\n", " wah", "wow", " dong", " sih", " deh", "teung"}

    for i:=0; i<len(replacableWords); i++{
        sentence = strings.Replace(sentence, replacableWords[i], "", -1)
    }

    regex := regexp.MustCompile("((wk)+(w?)+(k?)+)+")
    sentence = regex.ReplaceAllString(sentence, "")

    regex = regexp.MustCompile("((xi)+(x?)+(i?)+)+")
    sentence = regex.ReplaceAllString(sentence, "")

    regex = regexp.MustCompile("((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+")
    sentence = regex.ReplaceAllString(sentence, "")

    splittedSentence := strings.Split(sentence, " ")
    if splittedSentence[0] == "" {
        splittedSentence = splittedSentence[1:]
    }
    sentence = strings.Join(splittedSentence, " ")

    if strings.TrimSpace(sentence) != ""{
        sentence = strings.TrimSpace(sentence)
        splittedSentence = strings.Split(sentence, " ")
        normalSentence := " "
        for i:=0; i<len(splittedSentence); i++{
            normalizeWord := CheckNormalWord(splittedSentence[i])
            rootSentence := stemmer.Stem(normalizeWord)
            normalSentence += rootSentence + " "
        }
        return punct_re_escape.ReplaceAllString(normalSentence, "")
    }
    return sentence
}

func CheckNormalWord(word string) string {
    slangResult := DynamicSwitcher(dataSlang, word)
    if strings.TrimSpace(slangResult) != "" {
        return slangResult
    }
    return word
}

func DynamicSwitcher(dict map[string]string, key string) string {
    return dict[key]
}