from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#from optimum.onnxruntime import ORTModelForSequenceClassification

# use roberta finetuned for multiclass classification on go_emotions
model_id = "SamLowe/roberta-base-go_emotions-onnx"
file_name = "onnx/model_quantized.onnx"


tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(model_id)
#model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name)


roberta_pipeline = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    function_to_apply="sigmoid",  # optional as is the default for the task
)
