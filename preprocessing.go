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
    input := flag.String("i", "dataset/qa.csv", "input file directory")
    output := flag.String("o", "dataset/clean_qa.txt", "output file directory")
    flag.Parse()
    
    var questionLength = make(map[int]int)
    var answerLength = make(map[int]int)
    
    // open file
    f, err := os.Open(*input)
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

 
    f, err = os.Open(*input)
    reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1
    reader.Comma = '|'
    records, _ := reader.ReadAll()

    f, err = os.Create(*output)

    if err != nil {
        log.Fatal(err)
    }

    defer f.Close()

    for i, record := range records {
        if i == 0 {
            continue
        }

        question := record[0]
        answer := ""
        if len(record) > 1 {
            answer = record[1]
        }

        question = NormalizeSentence(question)
        question = NormalizeSentence(question)
        question = stemmer.Stem(question)

        answer = strings.ToLower(answer)
        answer = strings.Replace(answer, "iteung", "aku", -1)
        answer = strings.Replace(answer, "\n", " ", -1)
        if len(strings.Split(question, " ")) > 0 && len(strings.Split(question, " ")) < 13 && len(strings.Split(answer, " ")) < 29{
            _, err := f.WriteString(fmt.Sprintf("%s\n%s\n\n", strings.TrimSpace(question), answer))

            if err != nil {
                log.Fatal(err)
            }
        }
    }
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