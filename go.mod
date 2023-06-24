module github.com/fahri-r/iteung-go

go 1.20

replace gorgonia.org/gorgonia => ./gorgonia

replace github.com/owulveryck/lstm => ./lstm

require (
	github.com/RadhiFadlillah/go-sastrawi v0.0.0-20200621225627-3dd6e0e1ac00
	github.com/go-gota/gota v0.12.0
	github.com/owulveryck/lstm v0.0.0-20180406085902-1581884e9d2d
	gorgonia.org/gorgonia v0.9.17
)

require (
	golang.org/x/net v0.0.0-20210423184538-5f58ad60dda6 // indirect
	gonum.org/v1/gonum v0.9.1 // indirect
)
