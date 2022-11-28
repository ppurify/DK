using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.IO;

// 사용법
// 1. dirpath, dir에 대상 폴더의 경로
// 2. 대상 폴더에는 에셋번들 파일만 존재해야함(다른파일 섞여있어도 작동함.상관없음. 에러만 뜸)
// 3. play버튼으로 실행

public class LoadAssetBundle : MonoBehaviour {
    private static string dirpath = "C:/Users/USER/Desktop/20221127abtest/";
    public DirectoryInfo dir = new DirectoryInfo("C:/Users/USER/Desktop/20221127abtest");
    private const string AssetBundlePath = "Assets/Resources/AssetBundles";
    public GameObject targetObject;


    void Start(){
        foreach (FileInfo File in dir.GetFiles())
        {
            string totalpath = dirpath+File.Name;
            ImportModelMesh (totalpath);
        }
    }

    
    void ImportModelMesh(string totalpath)
        {
            // Create AssetBundles Folder
            if (!System.IO.Directory.Exists(AssetBundlePath))
            {
                System.IO.Directory.CreateDirectory(AssetBundlePath);
                Debug.Log("AssetBundles folder created");
            }
            
            // load asset bundle file
            var loadedAssetBundle = AssetBundle.LoadFromFile(totalpath);
            if (loadedAssetBundle == null)
            {
                Debug.Log("Failed to load AssetBundle!");
                return;
            }
            
            // save asset bundle as prefab
            GameObject[] allPrefabs = loadedAssetBundle.LoadAllAssets<GameObject>();

            if (allPrefabs.Length > 1)
            {
                Debug.Log("Warning: Contains multiple objects!");
            }
            
            // only fist prefab place in the scene
            targetObject = Instantiate(allPrefabs[0]);
        }


}

