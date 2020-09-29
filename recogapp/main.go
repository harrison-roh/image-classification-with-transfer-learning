package main

import (
	"flag"
	"log"

	"github.com/gin-gonic/gin"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/api"
	"github.com/harrison-roh/image-recognition-with-transfer-learning/recogapp/inference"
)

func main() {
	modelsPath := flag.String("models", "", "Model path for inference")
	flag.Parse()

	i, err := inference.New(inference.Config{
		ModelsPath: *modelsPath,
	})
	if err != nil {
		log.Print(err)
		return
	}

	r := gin.Default()
	r.MaxMultipartMemory = 8 << 20

	a := api.APIs{
		I: i,
	}

	inferenceGroup := r.Group("/inference")
	{
		inferenceGroup.GET("", a.ListInfer)
		inferenceGroup.GET(":model", a.ShowInfer)
		inferenceGroup.POST("", a.InferDefault)
		inferenceGroup.POST(":model", a.InferWithModel)
	}

	r.Run(":18080")
}
