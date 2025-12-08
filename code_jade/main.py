import sys
import os

# Adiciona o diretório raiz ao path para importar módulos corretamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_jade.core import CodeJadeAgent

def main():
    print("==========================================")
    print("🚀 CodeJade - AI Developer Agent (v1.0)")
    print("==========================================")
    
    try:
        agent = CodeJadeAgent()
        print(f"🔧 Modelo: {agent.cfg.get('groq_model')}")
        print(f"📂 Work Dir: {agent.cfg.get('work_dir')}")
        print("💡 Digite 'sair' para encerrar.\n")

        while True:
            user_input = input("\n👨‍💻 Você: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("👋 Até logo!")
                break
            
            print("🤖 CodeJade pensando...")
            response = agent.chat_loop(user_input)
            print(f"\n🤖 CodeJade: {response}")

    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")

if __name__ == "__main__":
    main()
