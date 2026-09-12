# Contabilidade da confirmação V4 interrompida

Status: **INCOMPLETA**. Este documento registra a contabilidade dos artefatos locais examinados, sem calcular acurácia, ganho ou segurança parcial. Não é um comprovante de conclusão da avaliação nem uma atestação independente da execução.

## Chamadas e cobertura persistida

| Medida | Valor verificado |
|---|---:|
| Desenvolvimento 01 | 312 chamadas persistidas |
| Desenvolvimento 02 | 312 chamadas persistidas |
| Confirmação | 421 chamadas concluídas e persistidas |
| Total dos três diretórios | 1.045 chamadas persistidas |
| Marcadores `processing task, is_child = 0` no log | 1.046 |
| Tarefas com timing final no log | 1.045 |
| Diferença entre o log e as chamadas persistidas | 1 chamada cancelada |
| Passos da confirmação salvos em traces | 173 de 512 previstos |
| Resumos completos de episódio/política | 21 de 64 previstos |

As 1.045 chamadas persistidas, na ordem desenvolvimento 01, desenvolvimento 02 e confirmação, coincidem sequencialmente com os tokens de entrada e saída das primeiras 1.045 tarefas do log. Não houve divergência de tokens. Os timings de processamento da entrada e geração também coincidem, com tolerância de 0,006 ms para o arredondamento do log.

Os 21 resumos completos representam dez pares completos e o braço `structured` do 11º episódio. O grupo parcial é `rcwta-test-78c3b2c6-a / summary`, com cinco passos salvos, índices 0–4. Os resumos completos contabilizam 406 chamadas; as 15 chamadas restantes estão nas traces desse grupo parcial.

## Consumo e durações observados

Os valores seguintes abrangem somente as 421 chamadas persistidas da confirmação. A chamada cancelada não tem resposta final registrada e fica fora dos totais. Consequentemente, os totais de consumo e tempo observado são limites inferiores do trabalho efetivamente iniciado; as estatísticas de latência descrevem apenas as chamadas concluídas.

| Medida | Valor |
|---|---:|
| Tokens de entrada | 646.649 |
| Tokens de saída | 76.320 |
| Soma de `wall_seconds` das chamadas | 1.353,636751900136 s |
| Latência média | 3,215289196912437 s |
| Latência mediana | 3,0145233999937773 s |
| Latência mínima | 0,9599608999851625 s |
| Latência máxima | 9,828963800013298 s |
| P95, método nearest-rank | 5,303741599986097 s |
| Soma de `step_seconds` dos 173 passos | 1.357,503604600119 s |
| Finalizações `stop` | 417 |
| Finalizações `length` | 4 |

A soma de durações dos passos não é a duração completa da campanha. Não foi reconstruído um tempo final de episódio para o grupo interrompido. Seus cinco passos persistidos contêm 15 chamadas, 21.449 tokens de entrada, 2.628 tokens de saída e 50,56150060007349 s de `wall_seconds` somados.

Todos os eventos `complete` examinados na confirmação registram o modelo `rcwt-local-qwen35-4b`, seed `20260911`, custo de API zero e zero caracteres de reasoning. O protocolo declara `local_only_no_paid_api`. Esses registros não medem eletricidade ou amortização do equipamento e não provam a inexistência de outras execuções fora do escopo examinado.

## Última chamada e cancelamento

A última chamada persistida é `memory:summary`, correspondente à task **197388**. Ela registra 666 tokens de entrada, 130 de saída, `finish_reason=stop` e 1,9547294999938458 s de `wall_seconds`. Seus timings aparecem nas linhas 7618–7620 de `runtime/server.stderr.log`.

A task adicional **197522** inicia na linha 7624. A linha 7625 registra progresso `n_gen = 236`; a linha 7626 registra explicitamente `cancel task, id_task = 197522`. Não há timing final nem resposta persistida dessa task. O contador intermediário não foi convertido em um total final de tokens ou somado ao consumo das respostas concluídas.

As traces seguem exatamente o prefixo da ordem congelada. A próxima identidade prevista seria `rcwta-test-78c3b2c6-a / summary / step_index=5`. Associar a task cancelada a esse passo é uma inferência pela sequência, não uma identidade declarada pelo log. Estes registros não estabelecem quem ou por que interrompeu o processo.

## Integridade e limites da verificação

- O hash do protocolo coincide com `freeze.json` e `started.json`; o hash do schedule coincide com o freeze.
- Os hashes canônicos de `public.json` e `oracle.json` coincidem com o protocolo.
- Os 15 arquivos de fontes congeladas e os 15 arquivos correspondentes nas fontes atuais coincidiram com os hashes declarados no momento da inspeção.
- As hashes individuais das 173 traces e sua cadeia conferem; identidades e ordem correspondem ao prefixo congelado.
- Os hashes do protocolo e do completion do desenvolvimento selecionado coincidem com o vínculo declarado na confirmação.
- O diretório da confirmação examinado continha 8 arquivos imediatos, 15 arquivos em `sources` e 49 arquivos recursivos, incluindo a cópia do desenvolvimento.
- A confirmação não continha `completion.json`, `aborted.json` ou `partial-step.json`.

A verificação comparou conteúdo e hashes por leitura local. Não executou inferência, regeneração de corpus, replay semântico ou verificadores que exigem completion. Nenhum script novo de contabilidade foi salvo: os cálculos foram feitos em comandos de inspeção transitórios, sem importar os módulos do experimento. Esta nota não substitui um verificador reexecutável do pacote parcial.

## Hashes dos artefatos examinados

| Artefato | SHA-256 |
|---|---|
| `protocol.json` | `b09076db9d9311e7699d072a223bbd62d95b112fbb7973bbab87c950af101654` |
| `schedule.json` | `19673db3abafebea623772acdc0c54208d9080519a32a334f18aa748c156952a` |
| `traces.jsonl` | `beb01cde1aa808cc2204344bd9adef27fc0ac7ac10f0da8434c14501b2607676` |
| `episodes.jsonl` | `3a450546223c188c2cd139dcc418e0a91ddc4e39b7858100888c3a67a0183d4f` |
| `runtime/server.stderr.log` | `cd8f2cd409a502936db005d3d31360f0c81ce3ffd88740f28d20b95e5fbb5785` |

Os hashes de public/oracle no protocolo usam JSON canônico, e não os bytes dos arquivos formatados. O hash canônico verificado de public é `03c03e9e435d0223167fce9ea896dbd6b12686a6354a8c2b60c2c4d8ea6b6c98`; o de oracle é `4c602e2579d516533fd63959b37edd74c18804df6e84ddada7cd1dd4402a99f5`.

A integridade desse prefixo e sua conciliação com o log não autorizam declarar a confirmação concluída ou divulgar uma estimativa de ganho com o lote parcial.
