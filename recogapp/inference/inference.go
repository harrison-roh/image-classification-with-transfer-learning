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
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"gopkg.in/yaml.v2"
)

// Config 이미지 추론 모델 생성 설정정보
type Config struct {
	ModelsPath    string
	UserModelPath string
	LHost         string
}

// Inference 이미지 추론 모델 관리
type Inference struct {
	models        map[string]*iModel
	mutex         sync.RWMutex
	modelsPath    string
	userModelPath string

	lHost string
}

type modelConfig struct {
	Name                string   `yaml:"name"`
	Tags                []string `yaml:"tags"`
	InputShape          []int32  `yaml:"input_shape"`
	InputOperationName  string   `yaml:"input_operation_name"`
	OutputOperationName string   `yaml:"output_operation_name"`
	LabelsFile          string   `yaml:"labels_file"`
	Description         string   `yaml:"description"`
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
			i.addModel(m)
		}
	}

	if i.userModelPath != "" {
		m := getNewModel("", i.userModelPath)
		if err := loadModel(m); err != nil {
			log.Printf("Fail to load user model(%s): %s", i.userModelPath, err)
		} else {
			i.addModel(m)
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
		result, err := i.CreateModel("default", "", "Default Model")
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

func (i *Inference) delModel(delM *iModel) error {
	if delM.refCount > 0 {
		return fmt.Errorf("Currently in use: %s (%d)", delM.name, delM.refCount)
	}

	if err := os.RemoveAll(delM.modelPath); err != nil {
		return err
	}

	delete(i.models, delM.name)

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

// CreateRequest TODO
type CreateRequest struct {
	// Image label for this model
	Tag string `json:"tag"`

	// Model meta information
	ModelPath   string `json:"modelPath"`
	ConfigFile  string `json:"configFile"`
	Description string `json:"desc"`
}

// CreateModel TODO
func (i *Inference) CreateModel(newModel, tag, desc string) (map[string]interface{}, error) {
	modelDir := fmt.Sprintf("%s-%s", newModel, uuid.New().String()[:8])
	modelPath := path.Join(i.modelsPath, modelDir)

	m := getNewModel(newModel, modelPath)
	i.mutex.Lock()
	// 새로운 모델 생성 및 로드 전 슬롯 선점
	if err := i.addModel(m); err != nil {
		i.mutex.Unlock()
		return nil, err
	}
	// 모델 로드가 완료되기전 삭제가 되지 않도록 참조카운터를 증가
	i.getModel(newModel)
	i.mutex.Unlock()

	configFile := path.Join(modelPath, "config.yaml")

	req := CreateRequest{
		Tag:         tag,
		ModelPath:   modelPath,
		ConfigFile:  configFile,
		Description: desc,
	}

	j, _ := json.Marshal(req)
	data := bytes.NewBuffer(j)

	url := fmt.Sprintf("http://%s/model/%s", i.lHost, newModel)
	res, err := http.Post(url, "application/json", data)
	if err != nil {
		i.mutex.Lock()
		i.delModelUncond(m)
		i.mutex.Unlock()
		return nil, err
	}
	defer res.Body.Close()

	var response map[string]interface{}
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		i.mutex.Lock()
		i.delModelUncond(m)
		i.mutex.Unlock()
		return nil, err
	}

	if err := loadModel(m); err != nil {
		i.mutex.Lock()
		i.delModelUncond(m)
		i.mutex.Unlock()
		return nil, err
	}

	i.putModel(m)
	return response, nil
}

// GetModels 이미지 추론 모델 목록 반환
func (i *Inference) GetModels() []string {
	i.mutex.RLock()
	defer i.mutex.RUnlock()

	var models []string
	for model := range i.models {
		models = append(models, model)
	}

	return models
}

// GetModel 이미지 추론 모델 정보 반환
func (i *Inference) GetModel(model string) map[string]interface{} {
	i.mutex.RLock()
	m := i.getModel(model)
	i.mutex.RUnlock()

	if m == nil {
		return nil
	}
	defer i.putModel(m)

	var status string
	switch m.status {
	case modelStatusReady:
		status = "ready"
	case modelStatusRun:
		status = "run"
	default:
		status = "unknown"
	}

	return map[string]interface{}{
		"model":          m.name,
		"refCount":       m.refCount,
		"status":         status,
		"inputOperator":  m.inputOp,
		"outputOperator": m.outputOp,
		"inputShape":     m.inputShape,
		"numberOfLables": m.nrLables,
		"description":    m.desc,
	}
}

// Infer 추론
func (i *Inference) Infer(model, image, format string, k int) ([]InferLabel, error) {
	i.mutex.RLock()
	m := i.getModel(model)
	i.mutex.RUnlock()

	if m == nil {
		return nil, fmt.Errorf("No such model: %s", model)
	}
	defer i.putModel(m)

	if m.status != modelStatusRun {
		return nil, fmt.Errorf("Not ready yet")
	}

	result, err := m.infer(image, format)
	if err != nil {
		return nil, err
	}

	var infers []InferLabel
	for idx, prob := range result {
		if idx >= len(m.labels) {
			break
		}

		infers = append(infers, InferLabel{
			Prob:  prob,
			Label: m.labels[idx],
		})
	}
	sort.Sort(sortByProb(infers))

	if k <= 0 {
		k = 5
	}

	return infers[:k], nil
}

const (
	modelStatusReady = iota
	modelStatusRun
)

// Model 이미지 추론 모델
type iModel struct {
	name      string
	modelPath string
	status    int32
	refCount  int32

	tfModel    *tf.SavedModel
	inputOp    string
	outputOp   string
	inputShape []int32

	imageDecoder map[string]imageDecode

	nrLables int
	labels   []string

	desc string
}

// 이미지 타입의 디코더
type imageDecode struct {
	graph   *tf.Graph
	session *tf.Session
	input   tf.Output
	output  tf.Output
}

func (m *iModel) infer(image, format string) ([]float32, error) {
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
			m.tfModel.Graph.Operation(m.inputOp).Output(0): inputImage,
		},
		[]tf.Output{
			m.tfModel.Graph.Operation(m.outputOp).Output(0),
		},
		nil,
	); err != nil {
		return nil, err
	}

	return results[0].Value().([][]float32)[0], nil
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
		decode  tf.Output
		session *tf.Session
		graph   *tf.Graph
		err     error
	)

	if decoder, ok := m.imageDecoder[format]; ok {
		return decoder, nil
	}

	scope := op.NewScope()
	input := op.Placeholder(scope, tf.String)

	if format == "jpg" || format == "jpeg" {
		decode = op.DecodeJpeg(scope, input, op.DecodeJpegChannels(3))
	} else if format == "png" {
		decode = op.DecodePng(scope, input, op.DecodePngChannels(3))
	} else {
		return imageDecode{}, fmt.Errorf("Unsupported image format: %s", format)
	}

	output := op.Div(scope,
		op.Sub(scope,
			op.ResizeBilinear(scope,
				op.ExpandDims(scope,
					op.Cast(scope, decode, tf.Float),
					op.Const(scope.SubScope("make_batch"), int32(0))),
				op.Const(scope.SubScope("size"), m.inputShape)),
			op.Const(scope.SubScope("mean"), float32(117))),
		op.Const(scope.SubScope("scale"), float32(1)))

	if graph, err = scope.Finalize(); err != nil {
		return imageDecode{}, err
	}

	if session, err = tf.NewSession(graph, nil); err != nil {
		return imageDecode{}, err
	}

	decoder := imageDecode{
		graph:   graph,
		input:   input,
		output:  output,
		session: session,
	}
	m.imageDecoder[format] = decoder

	return decoder, nil
}

func getNewModel(modelName, modelPath string) *iModel {
	return &iModel{
		name:      modelName,
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

	m.name = cfg.Name
	m.tfModel = tfModel
	m.inputOp = cfg.InputOperationName
	m.outputOp = cfg.OutputOperationName
	m.inputShape = cfg.InputShape[:2]
	m.imageDecoder = make(map[string]imageDecode)
	m.nrLables = len(labels)
	m.labels = labels
	m.desc = cfg.Description
	// Setting status should always be last
	m.status = modelStatusRun

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

// New 이미지 추론 모델 생성
func New(c Config) (i *Inference, err error) {
	i = &Inference{
		models:        make(map[string]*iModel),
		modelsPath:    c.ModelsPath,
		userModelPath: c.UserModelPath,
		lHost:         c.LHost,
	}
	err = i.init()

	return
}
