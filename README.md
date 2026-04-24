# Relatório Final

## Identificação

**Nome completo:** Jose Dhonantan Fernandes de Almeida

## Resumo da arquitetura do modelo

O modelo que foi implementado, é uma Rede Neural Convolucional (CNN) simples para classificação de dígitos do dataset MNIST.

### A arquitetura é composta por:

- Conv2D (32 filtros, 3x3, ReLU)
  - Responsável pela extração de características básicas como bordas e traços.
- MaxPooling2D (2x2)
  - Reduz a dimensionalidade espacial, diminuindo custo computacional e ajudando na generalização.
- Conv2D (64 filtros, 3x3, ReLU)
  - Aprende padrões mais complexos a partir das features iniciais.
- MaxPooling2D (2x2)
  - Nova redução dimensional.
- Flatten
  - Converte os mapas de características em um vetor unidimensional.
- Dense (64 neurônios, ReLU)
  - Combina as features extraídas pela CNN.
- Dense (10 neurônios, Softmax)
  - Camada de saída que gera a probabilidade para cada classe (0 a 9).
    - O modelo recebe imagens 28x28 em escala de cinza e retorna a probabilidade para cada uma das 10 classes.

## Principais bibliotecas utilizadas

- **TensorFlow** (incluindo Keras API)
- **os** (manipulação de arquivos)

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

O modelo atingiu aproximadamente, alta acurácia, **(≈99%)** no conjunto de testes
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
O treinamento foi realizado com poucas epocas (4), oque pode limitar desempenho máximo se os dados a serem treinados forem muito maiores

### Oque aprendi no desafio

- Pipeline completo: treino → avaliação → otimização → exportação
- Importância da otimização para uso em dispositivos embarcados
- Boas práticas na organização de scripts (treino e conversão separados)
