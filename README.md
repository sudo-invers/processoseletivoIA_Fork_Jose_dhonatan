# Relatório Final

## Identificação

**Nome completo:** Jose Dhonantan Fernandes de Almeida

## Resumo da arquitetura do modelo

O modelo que foi implementado, é uma Rede Neural Convolucional (CNN) simples para classificação de dígitos do dataset MNIST.

A arquitetura é composta por:

Uma camada **Conv2D** com 32 filtros e ativação ReLU (Unidade Linear Retificada) para extração inicial de características
Uma camada **MaxPooling2D** para redução de dimensionalidade
Uma segunda camada **Conv2D** com 64 filtros e ativação ReLU
Uma camada **MaxPooling2D**
Uma camada **Flatten** para transformar os dados em vetor
Uma camada **Dense** com 64 neurônios e ativação ReLU
Uma camada final **Dense** com 10 neurônios e ativação Softmax para classificação

O modelo recebe imagens 28x28 em escala de cinza e retorna a probabilidade para cada uma das 10 classes.

## Principais bibliotecas utilizadas

**TensorFlow** (incluindo Keras API)
**NumPy** (indiretamente via TensorFlow)
**os** (manipulação de arquivos)

## Técnica de otimização do modelo

utilizei a técnica de **Dynamic Range Quantization** durante a conversão do modelo para TensorFlow Lite.

Essa técnica:

Reduz o tamanho do modelo
Melhora a performance em dispositivos com recursos limitados
Mantém boa precisão ao converter pesos de ponto flutuante para representações mais compactas

A otimização foi aplicada com:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

## Resultados que foram obtidos

Após o treinamento do modelo:

O modelo atingiu aproximadamente, alta acurácia, **(≈99.8%)** no conjunto de testes
O modelo foi salvo em formato `.h5`
Após otimização, foi gerado um modelo `.tflite` mais leve e eficiente

## Comentários

### Dificuldades que encontrei

Configuração inicial do pipeline de conversão para TensorFlow Lite me foram complicadas, por não ter muita pratica com otimização de modelos
Garantir compatibilidade entre modelo Keras e TFLite demorei bastante para conseguir, a pesquisa pela documentação e exemplos do stack overflow me foram bem demoradas

### Decisões técnicas importantes

Uso de uma CNN simples para manter baixo custo computacional
Aplicação de quantização para melhorar eficiência em deploy

### Limitações do modelo

Arquitetura simples pode não capturar padrões muito mais complexos
O treinamento foi realizado com poucas epocas (5), oque pode limitar desempenho máximo

### Oque aprendi no desafio

Pipeline completo: treino → avaliação → otimização → exportação
Importância da otimização para uso em dispositivos embarcados
Boas práticas na organização de scripts (treino e conversão separados)
