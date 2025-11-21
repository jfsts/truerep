# TrueRep - Box Jump Counter

Aplicação de visão computacional para contagem de agachamentos usando MediaPipe.

## Funcionalidades

- Detecção de postura (Em Pé / Agachado)
- Contador de repetições automático
- Visualização em tempo real com skeleton
- Controles Start/Stop/Reset
- Suporte para câmera frontal em dispositivos móveis

## Tecnologias

- **Backend**: FastAPI + Python
- **Frontend**: JavaScript + MediaPipe Pose Landmarker
- **Visão Computacional**: MediaPipe Tasks Vision
- **Estilo**: Tailwind CSS

## Como usar

1. Acesse a aplicação
2. Permita acesso à câmera
3. Posicione-se de forma que seu corpo inteiro apareça na tela
4. Clique em "INICIAR"
5. Aguarde a mensagem "FIQUE EM PÉ" estabilizar
6. Comece a fazer agachamentos!

## Configurações

- **Ângulo de agachamento**: 90° (agachamento completo)
- **Ângulo em pé**: 165°
- **Tempo de estabilização**: 15 frames (~0.5s)
