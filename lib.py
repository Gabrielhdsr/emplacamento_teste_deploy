# lib.py (Adicione esta função)

def formatar_br(valor, casas_decimais: int = 0) -> str:
    """
    Formata um número para o padrão brasileiro (ponto para milhar, vírgula para decimal).
    """
    if valor is None:
        return "N/A"
    
    # 1. Converte o valor para string usando o padrão americano (vírgula como milhar)
    # Garante que o número de casas decimais seja respeitado
    s = f"{valor:,.{casas_decimais}f}"
    
    # 2. Trocas para o padrão BR:
    # Troca o separador de milhar (vírgula no americano) por um temporário
    s = s.replace(",", "X") 
    
    # Troca o separador decimal (ponto no americano) por vírgula (no BR)
    s = s.replace(".", ",") 
    
    # Troca o temporário de volta para ponto (separador de milhar no BR)
    s = s.replace("X", ".") 
    
    return s