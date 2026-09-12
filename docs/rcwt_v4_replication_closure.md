# R1: ganho medido com modelo local

Fechamento experimental em 12/09/2026. Uma nova confirmação autorizada, após
preservar a tentativa anterior interrompida. A publicação foi autorizada
separadamente depois desse fechamento; ela não modifica o experimento. O envio
de mensagens externas não faz parte da preparação do repositório.

## Resultado

O sistema de memória estruturada e organização de evidências públicas passou
no critério registrado: **+33,59 pontos percentuais** de acerto frente ao resumo
contínuo produzido pelo LLM, com **IC95% pareado [+27,34; +39,45] pp**.
São 32 episódios pareados, oito decisões por política por episódio, 512 decisões
no total e 1.248 chamadas a um Qwen3.5-4B local. O intervalo usa 10.000
reamostragens por episódio pareado, não 512 decisões independentes.

| Medida na confirmação completa | Resumo contínuo | Memória + contexto estruturados |
| --- | ---: | ---: |
| Ações exatamente corretas | 152/256 (59,38%) | 238/256 (92,97%) |
| Episódios inteiramente corretos | 2/32 | 19/32 |
| Tentativas inseguras fictícias | 11 | 4 |
| Centavos fictícios inseguros efetivados | 743.614 | 297.683 |
| Tokens de todas as gerações | 1.165.452 | 972.074 |
| Chamadas, incluindo compactação | 736 | 512 |
| Latência por passo: mediana / p95 | 10,460 / 14,936 s | 7,723 / 11,532 s |
| Tempo contabilizado dos episódios | 2.709,096 s | 2.071,715 s |

O candidato acertou 86 ações adicionais, utilizou 16,59% menos tokens e teve
mediana de tempo por passo 26,16% menor. Tokens e tempos incluem os dois passes
do ator e a compactação de memória; o passo também inclui tokenização e leitura
das evidências. São medições descritivas nesta máquina e carga, não promessas de
velocidade universal. API: US$0. Energia, depreciação e custo monetário total não
foram medidos; não são zero.

## O que foi realmente melhorado

O agente recebe observações públicas de ferramentas fictícias, gera um plano e
uma ação JSON, executa somente a ação final sobre um ledger simulado e carrega
apenas sua memória retida para o passo seguinte. O candidato organiza registros
por identificadores, preserva relações e atualizações e reúne memória com fatos
públicos do passo atual. Ambos os braços usam o mesmo ator congelado, modelo,
regras, schema de saída e limites de tokens.

O ganho pertence a esse **sistema combinado**, não apenas à compactação ou a
uma melhora intrínseca do LLM. O código não consulta o gabarito para escolher
ações, não corrige semanticamente as respostas e não repete chamadas até acertar.
A implementação não é aprendizado autônomo nem autoaperfeiçoamento recursivo.

As novas instâncias pertencem às mesmas quatro famílias sintéticas do gerador.
Não são tarefas de domínios novos, dados de clientes CloudWalk ou tráfego real.
Um modelo, uma seed de inferência e um conjunto confirmatório não estabelecem
generalização para outros modelos, seeds ou produção.

## Falhas que permanecem

Os dois limites **agregados e descritivos** de segurança passaram, mas não houve
ausência de regressões. Na família `insufficient-evidence`, tentativas inseguras
subiram de 1 para 2 e os centavos fictícios inseguros de 142.330 para 160.695.
Em `updated-state`, tentativas inseguras subiram de 0 para 1, sem dinheiro
inseguro efetivado nesse grupo. Truncamentos de memória aumentaram de 8 para 10.

O candidato teve 18 erros: sete adiamentos desnecessários, quatro execuções
inseguras, quatro decisões erradas e três motivos errados. Logo, **não está
certificado como seguro para produção**. Os critérios registrados não foram
alterados após ver esses resultados. O IC positivo confirma o efeito nesta
comparação; não prova que o ganho populacional mínimo seja de dez pontos.

## Evidência e verificações

- [Relatório completo](../results/agent_v4_replication/RESULTS.md),
  [análise](../results/agent_v4_replication/analysis.json) e
  [verificação](../results/agent_v4_replication/verification.json): replay offline
  integral de 512 decisões e 64 trajetórias; relatório recalculado e comparado
  byte a byte; critérios de acurácia e segurança agregada atendidos.
- [Registro prévio](rcwt_v4_replication_protocol.md): candidato e 15 fontes
  preservados desde o desenvolvimento selecionado; três fontes de orquestração
  e histórico também vinculados por hash antes da execução.
- [Conciliação das chamadas](../results/agent_v4_replication_accounting/CALL-ACCOUNTING.md):
  1.248 inícios, 1.248 pares finais de tempos e 1.248 chamadas registradas
  correspondentes, em ordem; zero chamadas extras ou canceladas. Total:
  2.137.526 tokens. A verificação dos bytes salvos também passou.
- Custódia privada do runtime (não distribuída no Git):
  apenas o servidor possuído PID 32156 foi encerrado, após conferir sua
  identidade e a conclusão do executor. Foram preservados 22 logs e recibos;
  nenhum arquivo de pesos ou histórico foi removido.
- A cópia final preserva os 32 arquivos originais (27.571.092 bytes), com hashes
  iguais antes e depois da cópia. Nenhum resultado foi sobrescrito.
- 583 testes Python passaram, além de compilação, `git diff --check` e 53
  verificações sintéticas do fechamento PowerShell. Os 18 hashes de fontes
  congeladas conferem. Resultados v2, v3, desenvolvimento v4 e prefixo
  interrompido foram novamente verificados offline e continuam separados.

O primeiro procedimento de fechamento encontrou um bloqueio de leitura do log
aberto no Windows, antes de criar a cópia ou parar o servidor. Corrigiu-se
somente o helper de custódia, fora das fontes congeladas. A segunda execução
encerrou o servidor e copiou os logs com acesso exclusivo e hashes conferidos;
a conciliação completa posterior confirmou os 1.248 registros. Nenhuma
inferência foi refeita por causa desse problema.

O log final tem SHA-256
`b138c28f609056c927d4c8e40fa4106ab85e5688e80e0d66080ef966c94c2c2d`.
O protocolo R1 tem SHA-256
`5fdb1fcb9b81bd4fc073d9429c90e199a44688dfbdf9b2bdfedc74f1c9092524`.
Hashes e telemetria locais permitem auditoria de consistência, não atestação
física independente ou prova de inexistência de execuções não listadas.

Consulte os [comandos de replay e conciliação](rcwt_v4_replication_demo.md).
Os logs brutos estão ignorados pelo Git e os recibos contêm caminhos locais;
a auditoria dos originais requer o bundle privado preservado. Para a publicação,
uma [cópia derivada](../results/agent_v4_replication_public_runtime/manifest.json)
remove apenas um prefixo de diretório da metadata, preservando todas as linhas
e eventos. Sua [conciliação pública](../results/agent_v4_replication_public_accounting/CALL-ACCOUNTING.md)
pode ser refeita sem o bundle privado. Ela não é o log original nem atestação
física independente. A autorização de publicação não apaga as limitações ou
transforma verificação de registros em uma nova confirmação experimental.

## Histórico preservado

A [confirmação anterior](../results/agent_v4_interrupted/INTERRUPTION.md) permanece
incompleta: 173/512 decisões, 421 chamadas persistidas e uma chamada cancelada
sem resposta ou uso final persistidos. Ela não foi retomada nem agrupada com R1.
Os custos anteriores e as duas tentativas de desenvolvimento permanecem
documentados separadamente. Resultados negativos anteriores não foram apagados.
