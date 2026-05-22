import sys
import os
from PyQt6.QtWidgets import QApplication, QDialog
from ui.main_window import LoginDialog, MainWindow

def main():
    # Inicialización de la aplicación Qt
    app = QApplication(sys.argv)
    
    # Crear carpeta de datos si no existe
    if not os.path.exists("data"):
        try:
            os.makedirs("data")
        except Exception:
            pass

    # Mostrar ventana flotante de Login para selección de Rol
    login = LoginDialog()
    if login.exec() == QDialog.DialogCode.Accepted:
        # Se extrae el rol seleccionado para ajustar el modo de visualización de la ventana principal
        user_role = login.selected_role
        print(f"[ACCESO] Iniciando aplicación en rol de: {user_role}")
        
        main_win = MainWindow(role=user_role)
        main_win.show()
        
        sys.exit(app.exec())
    else:
        print("[ACCESO] Inicio de sesión cancelado o denegado.")
        sys.exit(0)

if __name__ == "__main__":
    main()
