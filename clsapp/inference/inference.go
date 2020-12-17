package inference

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/google/uuid"
	"github.com/harrison-roh/image-classification-with-transfer-learning/clsapp/constants"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"gopkg.in/yaml.v2"
)

// Config 이미지 추론 모델 생성 설정정보
type Config struct {
	UserModelPath string
	LHost         string
}

// Inference 이미지 추론 모델 관리
type Inference struct {
	models        map[string]*iModel
	rwMutex       sync.RWMutex
	modelsPath    string
	userModelPath string

	lHost string
}

const (
	binaryClass = "binary"
	multiClass  = "multi"
)

type trainingResult struct {
	Epochs             int       `yaml:"epochs"`
	InitLoss           float32   `yaml:"initLoss"`
	InitAccuracy       float32   `yaml:"initAccuracy"`
	TrainLoss          []float32 `yaml:"trainLoss"`
	TrainAccuracy      []float32 `yaml:"trainAccuracy"`
	ValidationLoss     []float32 `yaml:"validationLoss"`
	ValidationAccuracy []float32 `yaml:"validationAccuracy"`
}

type modelConfig struct {
	Name                string         `yaml:"name"`
	Type                string         `yaml:"type"`
	Tags                []string       `yaml:"tags"`
	Classification      string         `yaml:"classification"`
	InputShape          []int32        `yaml:"inputShape"`
	InputOperationName  string         `yaml:"inputOperationName"`
	OutputOperationName string         `yaml:"outputOperationName"`
	LabelsFile          string         `yaml:"labelsFile"`
	TrainingResult      trainingResult `yaml:"trainingResult"`
	Description         string         `yaml:"description"`
}

func (i *Inference) loadModels() error {
	dirs, _ := ioutil.ReadDir(i.modelsPath)

	for _, dir := range dirs {
		modelPath := path.Join(i.modelsPath, dir.Name())

		m := getNewModel("", modelPath)
		if err := loadModel(m); err != nil {
			log.Printf("Fail to load model(%s): %s", modelPath, err)
			i.delModelUncond(m)
		} else {
			if err := i.addModel(m); err != nil {
				log.Print(err)
			}
		}
	}

	if i.userModelPath != "" {
		m := getNewModel("", i.userModelPath)
		if err := loadModel(m); err != nil {
			log.Printf("Fail to load user model(%s): %s", i.userModelPath, err)
		} else {
			if err := i.addModel(m); err != nil {
				log.Print(err)
			}
		}
	}

	return nil
}

func (i *Inference) init() error {
	if err := i.loadModels(); err != nil {
		return err
	}

	if len(i.models) == 0 {
		// 아무런 추론 모델이 없는 경우 기본 모델을 생성
		result, err := i.CreateModel(
			constants.DefaultModelName,
			"",
			"Default Model",
			constants.TrainEpochs,
			false)
		if err != nil {
			return err
		}
		log.Printf("Create default model: %v", result)
	}

	return nil
}

func (i *Inference) addModel(newM *iModel) error {
	if newM.name == "" {
		return errors.New("Empty model name")
	}

	for model, m := range i.models {
		if model == newM.name || m.name == newM.name {
			return fmt.Errorf("Duplicated model: %s", newM.name)
		} else if m.modelPath == newM.modelPath {
			return fmt.Errorf("Duplicated model path: %s", newM.modelPath)
		}
	}

	i.models[newM.name] = newM
	return nil
}

func (i *Inference) delModel(model string) error {
	m, ok := i.models[model]
	if !ok {
		return fmt.Errorf("No such model: %s", model)
	}

	if m.refCount > 0 {
		return fmt.Errorf("Currently in use: %s (%d)", m.name, m.refCount)
	}

	if err := os.RemoveAll(m.modelPath); err != nil {
		return err
	}

	delete(i.models, m.name)

	return nil
}

func (i *Inference) delModelUncond(delM *iModel) {
	if err := os.RemoveAll(delM.modelPath); err != nil {
		log.Print(err)
	}

	delete(i.models, delM.name)
}

func (i *Inference) getModel(model string) *iModel {
	if m, ok := i.models[model]; ok {
		atomic.AddInt32(&m.refCount, 1)
		return m
	}

	return nil
}

func (i *Inference) putModel(m *iModel) {
	atomic.AddInt32(&m.refCount, -1)
}

// CreateRequest 모델 생성 요청
type CreateRequest struct {
	// Image root path for training
	ImagePath string `json:"imagePath"`

	// Model meta information
	ModelPath   string `json:"modelPath"`
	ConfigFile  string `json:"configFile"`
	Description string `json:"desc"`

	Epochs int `json:"epochs"`

	Trial bool `json:"trial"`
}

// CreateResponse 모델 생성 응답
type CreateResponse struct {
	ModelPath string `json:"modelPath" binding:"required"`
}

// CreateModel 추론모델 생성
func (i *Inference) CreateModel(newModel, subject, desc string, epochs int, trial bool) (map[string]interface{}, error) {
	modelDir := fmt.Sprintf("%s-%s", newModel, uuid.New().String()[:8])
	modelPath := path.Join(i.modelsPath, modelDir)

	m := getNewModel(newModel, modelPath)
	i.rwMutex.Lock()
	// 새로운 모델 생성 및 로드 전 슬롯 선점
	if err := i.addModel(m); err != nil {
		i.rwMutex.Unlock()
		return nil, err
	}
	i.getModel(newModel)
	i.rwMutex.Unlock()
	defer i.putModel(m)

	configFile := path.Join(modelPath, "config.yaml")
	imagePath := ""
	if subject != "" {
		imagePath = path.Join(constants.ImagesPath, subject)
	}

	req := CreateRequest{
		ImagePath:   imagePath,
		ModelPath:   modelPath,
		ConfigFile:  configFile,
		Description: desc,
		Epochs:      epochs,
		Trial:       trial,
	}

	j, _ := json.Marshal(req)
	data := bytes.NewBuffer(j)

	url := fmt.Sprintf("http://%s/models/%s", i.lHost, newModel)
	res, err := http.Post(url, "application/json", data)
	if err != nil {
		i.rwMutex.Lock()
		i.delModelUncond(m)
		i.rwMutex.Unlock()
		return nil, err
	}
	defer res.Body.Close()

	var response map[string]interface{}
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		i.rwMutex.Lock()
		i.delModelUncond(m)
		i.rwMutex.Unlock()
		return nil, err
	}

	atomic.StoreInt32(&m.status, modelStatusBuild)

	return response, nil
}

// OperateModel 생성 된 추론모델 로드
func (i *Inference) OperateModel(model, modelPath string) error {
	i.rwMutex.RLock()
	m := i.getModel(model)
	i.rwMutex.RUnlock()

	if m == nil {
		if err := os.RemoveAll(modelPath); err != nil {
			log.Print(err)
		}
		return fmt.Errorf("No such model for register: %s", model)
	}
	defer i.putModel(m)

	if m.modelPath != modelPath {
		i.rwMutex.Lock()
		i.delModelUncond(m)
		i.rwMutex.Unlock()
		return fmt.Errorf("Invalid model path: %s", model)
	}

	if err := loadModel(m); err != nil {
		i.rwMutex.Lock()
		i.delModelUncond(m)
		i.rwMutex.Unlock()
		return err
	}

	return nil
}

// DeleteModel 모델 삭제
func (i *Inference) DeleteModel(model string) error {
	i.rwMutex.Lock()
	defer i.rwMutex.Unlock()

	return i.delModel(model)
}

// GetModels 이미지 추론 모델 목록 반환
func (i *Inference) GetModels() []string {
	i.rwMutex.RLock()
	defer i.rwMutex.RUnlock()

	var models []string
	for model := range i.models {
		models = append(models, model)
	}

	return models
}

// GetModel 이미지 추론 모델 정보 반환
func (i *Inference) GetModel(model string, verbose bool) map[string]interface{} {
	i.rwMutex.RLock()
	m := i.getModel(model)
	i.rwMutex.RUnlock()

	if m == nil {
		return nil
	}
	defer i.putModel(m)

	var status string
	switch atomic.LoadInt32(&m.status) {
	case modelStatusReady:
		status = "ready"
	case modelStatusBuild:
		status = "build"
	case modelStatusRun:
		status = "run"
	default:
		status = "unknown"
	}

	var labels []string
	if verbose {
		labels = make([]string, len(m.labels))
		copy(labels, m.labels)
	} else {
		l := 10
		if l > len(m.labels) {
			l = len(m.labels)
		}
		labels = make([]string, l)
		copy(labels, m.labels)
		if len(m.labels) > l {
			labels = append(labels, "...")
		}
	}

	info := map[string]interface{}{
		"model":          m.name,
		"refCount":       m.refCount,
		"inputShape":     m.inputShape,
		"numberOfLables": m.nrLables,
		"type":           m.cfg.Type,
		"classification": m.cfg.Classification,
		"inputOperator":  m.cfg.InputOperationName,
		"outputOperator": m.cfg.OutputOperationName,
		"description":    m.cfg.Description,
		"status":         status,
		"lables":         labels,
	}

	if verbose {
		trainingInfo := map[string]interface{}{
			"epochs":             m.cfg.TrainingResult.Epochs,
			"initLoss":           m.cfg.TrainingResult.InitLoss,
			"initAccuracy":       m.cfg.TrainingResult.InitAccuracy,
			"trainLoss":          m.cfg.TrainingResult.TrainLoss,
			"trainAccuracy":      m.cfg.TrainingResult.TrainAccuracy,
			"validationLoss":     m.cfg.TrainingResult.ValidationLoss,
			"validationAccuracy": m.cfg.TrainingResult.ValidationAccuracy,
		}

		info["trainingResult"] = trainingInfo

	}

	return info
}

// Infer 추론
func (i *Inference) Infer(model, image, format string, k int) ([]InferLabel, error) {
	i.rwMutex.RLock()
	m := i.getModel(model)
	i.rwMutex.RUnlock()

	if m == nil {
		return nil, fmt.Errorf("No such model: %s", model)
	}
	defer i.putModel(m)

	if atomic.LoadInt32(&m.status) != modelStatusRun {
		return nil, fmt.Errorf("Not ready yet")
	}

	return m.infer(image, format, k)
}

const (
	modelStatusReady = iota
	modelStatusBuild
	modelStatusRun
)

// Model 이미지 추론 모델
type iModel struct {
	name      string
	modelPath string
	cfg       modelConfig
	status    int32
	refCount  int32

	tfModel    *tf.SavedModel
	inputShape []int32

	imageDecoder map[string]imageDecode
	idMutex      sync.Mutex

	nrLables int
	labels   []string
}

// 이미지 타입의 디코더
type imageDecode struct {
	graph   *tf.Graph
	session *tf.Session
	input   tf.Output
	output  tf.Output
}

func (m *iModel) infer(image, format string, k int) ([]InferLabel, error) {
	var (
		inputImage *tf.Tensor
		results    []*tf.Tensor
		err        error
	)

	if inputImage, err = m.normInputImage(image, format); err != nil {
		return nil, err
	}

	if results, err = m.tfModel.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.tfModel.Graph.Operation(m.cfg.InputOperationName).Output(0): inputImage,
		},
		[]tf.Output{
			m.tfModel.Graph.Operation(m.cfg.OutputOperationName).Output(0),
		},
		nil,
	); err != nil {
		return nil, err
	}

	probabilities := results[0].Value().([][]float32)[0]

	if m.cfg.Classification == binaryClass {
		return m.classifyBinary(probabilities[0])
	} else if m.cfg.Classification == multiClass {
		return m.classifyMulti(probabilities, k)
	}

	return nil, fmt.Errorf("Unknown classification: %s", m.cfg.Classification)
}

func (m *iModel) normInputImage(image, format string) (*tf.Tensor, error) {
	var (
		decoder     imageDecode
		imageTensor *tf.Tensor
		norms       []*tf.Tensor
		err         error
	)

	if decoder, err = m.getImageDecoder(format); err != nil {
		return nil, err
	}

	if imageTensor, err = tf.NewTensor(image); err != nil {
		return nil, err
	}

	if norms, err = decoder.session.Run(
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

func (m *iModel) getImageDecoder(format string) (imageDecode, error) {
	var (
		decoder imageDecode
		decode  tf.Output
		session *tf.Session
		graph   *tf.Graph
		ok      bool
		err     error
	)

	// 생성 된 디코더는 공용으로 사용되기 때문에,
	// 최초 생성시 lock을 잡도록 하고 이 후 사용할땐 lock 없이 접근
	decoder, ok = m.imageDecoder[format]
	if ok {
		return decoder, nil
	}

	m.idMutex.Lock()
	defer m.idMutex.Unlock()

	decoder, ok = m.imageDecoder[format]
	if ok {
		return decoder, nil
	}

	scope := op.NewScope()
	input := op.Placeholder(scope, tf.String)

	if format == "jpg" || format == "jpeg" {
		decode = op.DecodeJpeg(scope, input, op.DecodeJpegChannels(3))
	} else if format == "png" {
		decode = op.DecodePng(scope, input, op.DecodePngChannels(3))
	} else {
		return decoder, fmt.Errorf("Unsupported image format: %s", format)
	}

	// TODO 모델에 따라 이미지값 범위 조정
	// [0, 255]의 이미지값을 [-1, 1]로 조정: (image / 127.5) - 1
	normalizer := op.Sub(scope,
		op.Div(scope, op.Cast(scope, decode, tf.Float), op.Const(scope.SubScope("scale"), float32(127.5))),
		op.Const(scope.SubScope("offset"), float32(1)))

	// 임의의 크기(height, width) 이미지를 입력 크기(inputShape,)로 조정
	output := op.ResizeBilinear(scope,
		op.ExpandDims(scope, normalizer, op.Const(scope.SubScope("batch"), int32(0))),
		op.Const(scope.SubScope("resize"), m.inputShape))

	if graph, err = scope.Finalize(); err != nil {
		return decoder, err
	}

	if session, err = tf.NewSession(graph, nil); err != nil {
		return decoder, err
	}

	decoder = imageDecode{
		graph:   graph,
		input:   input,
		output:  output,
		session: session,
	}
	m.imageDecoder[format] = decoder

	return decoder, nil
}

func (m *iModel) classifyBinary(prob float32) ([]InferLabel, error) {
	var (
		idx    int
		infers []InferLabel
	)

	idx = 0
	if prob >= 0.5 {
		idx = 1
	} else {
		prob = 1 - prob
	}

	infers = make([]InferLabel, 1)
	infers[0].Prob = prob
	infers[0].Label = m.labels[idx]

	return infers, nil
}

func (m *iModel) classifyMulti(probs []float32, k int) ([]InferLabel, error) {
	if len(probs) != m.nrLables {
		return nil, fmt.Errorf(
			"The number of correct(%d) and predicted(%d) labels does not match",
			m.nrLables,
			len(probs),
		)
	}

	var infers []InferLabel
	for idx, prob := range probs {
		infers = append(infers, InferLabel{
			Prob:  prob,
			Label: m.labels[idx],
		})
	}
	sort.Sort(sortByProb(infers))

	if k <= 0 {
		k = constants.DefaultMultiClassMax
	}

	if k > len(infers) {
		k = len(infers)
	}

	return infers[:k], nil
}

func getNewModel(model, modelPath string) *iModel {
	return &iModel{
		name:      model,
		modelPath: modelPath,
		status:    modelStatusReady,
	}
}

func loadModel(m *iModel) error {
	var (
		cfgBytes []byte
		cfg      modelConfig
		tfModel  *tf.SavedModel
		labelsFp *os.File
		labels   []string
		err      error
	)

	// config 로드
	cfgFile := path.Join(m.modelPath, "config.yaml")
	if cfgBytes, err = ioutil.ReadFile(cfgFile); err != nil {
		return err
	}

	if err := yaml.Unmarshal(cfgBytes, &cfg); err != nil {
		return err
	}

	if m.name != "" && m.name != cfg.Name {
		return fmt.Errorf("Not matched model name[%s] in configuration[%s]", m.name, cfg.Name)
	}

	// model 로드
	if tfModel, err = tf.LoadSavedModel(m.modelPath, cfg.Tags, nil); err != nil {
		return err
	}

	// labels 로드
	labelsFile := path.Join(m.modelPath, cfg.LabelsFile)
	if labelsFp, err = os.Open(labelsFile); err != nil {
		return err
	}
	defer labelsFp.Close()

	scanner := bufio.NewScanner(labelsFp)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	m.cfg = cfg
	m.name = cfg.Name
	m.tfModel = tfModel
	m.inputShape = cfg.InputShape[:2]
	m.imageDecoder = make(map[string]imageDecode)
	m.nrLables = len(labels)
	m.labels = labels
	// Setting status should always be last
	atomic.StoreInt32(&m.status, modelStatusRun)

	return nil
}

// InferLabel 이미지 추론 항목
type InferLabel struct {
	Prob  float32 `json:"probability"`
	Label string  `json:"label"`
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

// New 이미지 추론 모델 생성
func New(c Config) (i *Inference, err error) {
	i = &Inference{
		models:        make(map[string]*iModel),
		modelsPath:    constants.ModelsPath,
		userModelPath: c.UserModelPath,
		lHost:         c.LHost,
	}
	err = i.init()

	return
}
