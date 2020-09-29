package inference

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Config 이미지 추론 모델 생성 설정정보
type Config struct {
	ModelsPath string
}

// Inference 이미지 추론 모델 관리
type Inference struct {
	models     map[string]iModel
	modelsPath string

	idLock      sync.RWMutex
	imageDecode map[string]imageDecoder

	height, width int32
	mean          float32
	scale         float32
}

// Model 이미지 추론 모델
type iModel struct {
	graph   *tf.Graph
	session *tf.Session
	labels  []string
	desc    string
}

// 이미지 타입의 디코더
type imageDecoder struct {
	graph  *tf.Graph
	input  tf.Output
	output tf.Output
}

func loadModel(mFile, lFile, dFile string) (iModel, error) {
	var (
		graph        *tf.Graph
		session      *tf.Session
		labels       []string
		descriptions []string
		lFp          *os.File
		dFp          *os.File
		mByte        []byte
		err          error
	)

	// model 로드
	if mByte, err = ioutil.ReadFile(mFile); err != nil {
		return iModel{}, fmt.Errorf("Fail to read model: %s: %s", mFile, err)
	}

	graph = tf.NewGraph()
	if err := graph.Import(mByte, ""); err != nil {
		return iModel{}, fmt.Errorf("Fail to import model: %s", err)
	}

	if session, err = tf.NewSession(graph, nil); err != nil {
		return iModel{}, fmt.Errorf("Fail to make model session: %s", err)
	}

	// labels 로드
	if lFp, err = os.Open(lFile); err != nil {
		return iModel{}, fmt.Errorf("Fail to open label: %s: %s", lFile, err)
	}
	defer lFp.Close()

	lScanner := bufio.NewScanner(lFp)
	for lScanner.Scan() {
		labels = append(labels, lScanner.Text())
	}
	if err := lScanner.Err(); err != nil {
		return iModel{}, fmt.Errorf("Fail to read label: %s", err)
	}

	// description 로드
	if dFp, err = os.Open(dFile); err != nil {
		return iModel{}, fmt.Errorf("Fail to open description: %s: %s", dFile, err)
	}
	defer dFp.Close()

	dScanner := bufio.NewScanner(dFp)
	for dScanner.Scan() {
		descriptions = append(descriptions, dScanner.Text())
	}
	if err := dScanner.Err(); err != nil {
		return iModel{}, fmt.Errorf("Fail to read description: %s", err)
	}

	return iModel{
		graph:   graph,
		session: session,
		labels:  labels,
		desc:    strings.Join(descriptions, "\n"),
	}, nil
}

func (i *Inference) loadModels() error {
	models, err := ioutil.ReadDir(i.modelsPath)
	if err != nil {
		return err
	}

	for _, model := range models {
		modelPath := path.Join(i.modelsPath, model.Name())

		files, err := ioutil.ReadDir(modelPath)
		if err != nil {
			log.Print(err)
			continue
		}

		mFile := ""
		lFile := ""
		dFile := ""
		for _, file := range files {
			ext := filepath.Ext(file.Name())
			if ext == "" {
				continue
			}
			ext = ext[1:]

			if ext == "pb" {
				mFile = path.Join(modelPath, file.Name())
			} else if ext == "labels" {
				lFile = path.Join(modelPath, file.Name())
			} else if ext == "desc" {
				dFile = path.Join(modelPath, file.Name())
			}
		}

		if mFile == "" || lFile == "" || dFile == "" {
			log.Printf("%s model has not enough information (model=%s, lables=%s, desc=%s)",
				model.Name(), mFile, lFile, dFile)
			continue
		}

		if m, err := loadModel(mFile, lFile, dFile); err != nil {
			log.Print(err)
		} else {
			log.Printf("Model successfully loaded: %s", model.Name())
			i.models[model.Name()] = m
		}
	}

	return nil
}

func (i *Inference) getImageDecoder(format string) (imageDecoder, error) {
	var (
		decode tf.Output
	)

	i.idLock.RLock()
	if decoder, ok := i.imageDecode[format]; ok {
		i.idLock.RUnlock()
		return decoder, nil
	}
	i.idLock.RUnlock()

	i.idLock.Lock()
	defer i.idLock.Unlock()

	if decoder, ok := i.imageDecode[format]; ok {
		return decoder, nil
	}

	scope := op.NewScope()
	input := op.Placeholder(scope, tf.String)

	if format == "jpg" || format == "jpeg" {
		decode = op.DecodeJpeg(scope, input, op.DecodeJpegChannels(3))
	} else if format == "png" {
		decode = op.DecodePng(scope, input, op.DecodePngChannels(3))
	} else {
		return imageDecoder{}, fmt.Errorf("Unsupported image format: %s", format)
	}

	output := op.Div(scope,
		op.Sub(scope,
			// 이중선형보간법을 이용하여 이미지를 224x224 크기로 리사이징
			op.ResizeBilinear(scope,
				// 단일 이미지를 포함하는 배치 작업 생성
				op.ExpandDims(scope,
					// 디코딩 된 픽셀값 이용
					op.Cast(scope, decode, tf.Float),
					op.Const(scope.SubScope("make_batch"), int32(0))),
				op.Const(scope.SubScope("size"), []int32{i.height, i.width})),
			op.Const(scope.SubScope("mean"), i.mean)),
		op.Const(scope.SubScope("scale"), i.scale))

	graph, err := scope.Finalize()

	if err != nil {
		return imageDecoder{}, err
	}

	decoder := imageDecoder{
		graph:  graph,
		input:  input,
		output: output,
	}
	i.imageDecode[format] = decoder

	return decoder, nil
}

func (i *Inference) getNormInputImage(image, format string) (*tf.Tensor, error) {
	var (
		decoder     imageDecoder
		imageTensor *tf.Tensor
		session     *tf.Session
		norms       []*tf.Tensor
		err         error
	)

	if decoder, err = i.getImageDecoder(format); err != nil {
		return nil, err
	}

	if imageTensor, err = tf.NewTensor(image); err != nil {
		return nil, err
	}

	if session, err = tf.NewSession(decoder.graph, nil); err != nil {
		return nil, err
	}
	defer session.Close()

	if norms, err = session.Run(
		map[tf.Output]*tf.Tensor{
			decoder.input: imageTensor,
		},
		[]tf.Output{
			decoder.output,
		},
		nil,
	); err != nil {
		return nil, err
	}

	return norms[0], nil
}

// GetModels 이미지 추론 모델 목록 반환
func (i *Inference) GetModels() []string {
	var models []string
	for model := range i.models {
		models = append(models, model)
	}

	return models
}

// GetModel 이미지 추론 모델 정보 반환
func (i *Inference) GetModel(model string) map[string]interface{} {
	if m, ok := i.models[model]; ok {
		return map[string]interface{}{
			"Number of lables": len(m.labels),
			"Description":      m.desc,
		}
	}

	return nil
}

// InferLabel 이미지 추론 항목
type InferLabel struct {
	Prob  float32
	Label string
}

type sortByProb []InferLabel

func (s sortByProb) Len() int {
	return len(s)
}

func (s sortByProb) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s sortByProb) Less(i, j int) bool {
	return s[i].Prob > s[j].Prob
}

// Infer 추론
func (i *Inference) Infer(model, image, format string, k int) ([]InferLabel, error) {
	var (
		inputImage *tf.Tensor
		results    []*tf.Tensor
		err        error
	)

	m, ok := i.models[model]
	if !ok {
		return nil, fmt.Errorf("Cannot find model: %s", model)
	}

	if inputImage, err = i.getNormInputImage(image, format); err != nil {
		return nil, err
	}

	if results, err = m.session.Run(
		map[tf.Output]*tf.Tensor{
			m.graph.Operation("input").Output(0): inputImage,
		},
		[]tf.Output{
			m.graph.Operation("output").Output(0),
		},
		nil,
	); err != nil {
		return nil, err
	}

	probs := results[0].Value().([][]float32)[0]

	var infers []InferLabel
	for idx, p := range probs {
		if idx >= len(m.labels) {
			break
		}

		infers = append(infers, InferLabel{
			Prob:  p,
			Label: m.labels[idx],
		})
	}
	sort.Sort(sortByProb(infers))

	if k <= 0 {
		k = 5
	}

	return infers[:k], nil
}

// New 이미지 추론 모델 생성
func New(c Config) (i *Inference, err error) {
	i = &Inference{
		models:     make(map[string]iModel),
		modelsPath: c.ModelsPath,

		imageDecode: make(map[string]imageDecoder),

		height: 224,
		width:  224,
		mean:   float32(117),
		scale:  float32(1),
	}
	err = i.loadModels()

	return
}
