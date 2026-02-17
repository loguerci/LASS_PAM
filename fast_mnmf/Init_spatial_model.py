def init_spatial_model(self):
        """
        Initialisation directe pour cas surdéterminé (N <= M).
        Q = user_A^-1
        G = Diagonale (Source n -> Colonne n)
        """

        print(">>> Init Spatial: Inversion de user_A (Cas surdéterminé).")

        # 1. Calcul de Q 
        A_matrix = self.xp.asarray(self.user_A, dtype=self.TYPE_COMPLEX)
        self.Q_FMM = self.xp.linalg.inv(A_matrix)

        # 2. Initialisation de G 
        # On crée une matrice (N x M) remplie de g_eps
        self.G_NM = self.xp.full((self.n_source, self.n_mic), self.g_eps, dtype=self.TYPE_FLOAT)

        # 3. Activation directe (Source n utilise la Colonne n)
        # On place des 1 sur la diagonale (n_source x n_source)
        indices = self.xp.arange(self.n_source)
        self.G_NM[indices, indices] = 1.0

        # 4. Normalisation (Somme de chaque ligne de G = 1)
        self.G_NM /= self.G_NM.sum(axis=1, keepdims=True)

        # 5. Normalisation interne (W, H)
        self.normalize()