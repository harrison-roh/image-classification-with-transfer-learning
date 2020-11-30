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
	userModelPath := flag.String("usermodel", "", "Path for user inference model")
	learnHost := flag.String("learnhost", "learnapp:18090", "Model learning host")
	flag.Parse()

	i, err := inference.New(inference.Config{
		UserModelPath: *userModelPath,
		LHost:         *learnHost,
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
		inferenceGroup.POST("", a.InferDefault)
		inferenceGroup.POST(":model", a.InferWithModel)
	}

	modelsGroup := r.Group("/models")
	{
		modelsGroup.GET("", a.ListModels)
		modelsGroup.GET(":model", a.ShowModel)
		modelsGroup.POST(":model", a.CreateModel)
		modelsGroup.DELETE(":model", a.DeleteModel)
	}

	imagesGroup := r.Group("/images")
	{
		imagesGroup.GET("", a.ListImages)
		imagesGroup.POST("", a.UploadImages)
		imagesGroup.DELETE("", a.DeleteImages)
	}

	r.Run(":18080")
}
