package main

import (
	"flag"
	"log"

	"github.com/gin-gonic/gin"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/api"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/data"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/inference"
)

func main() {
	modelsPath := flag.String("models", "", "Model path for inference")
	flag.Parse()

	i, err := inference.New(inference.Config{
		ModelsPath: *modelsPath,
	})
	if err != nil {
		log.Fatal(err)
	}

	m, err := data.New()
	if err != nil {
		log.Fatal(err)
	}
	defer m.Destroy()

	r := gin.Default()
	r.MaxMultipartMemory = 8 << 20

	a := api.APIs{
		I: i,
		M: m,
	}

	inferenceGroup := r.Group("/inference")
	{
		inferenceGroup.GET("", a.ListInference)
		inferenceGroup.GET(":model", a.ShowInference)
		inferenceGroup.POST("", a.InferDefault)
		inferenceGroup.POST(":model", a.InferWithModel)
	}

	imageGroup := r.Group("/image")
	{
		imageGroup.POST("upload", a.UploadImage)
		imageGroup.POST("uploadMultiple", a.UploadImages)
	}

	r.Run(":18080")
}
