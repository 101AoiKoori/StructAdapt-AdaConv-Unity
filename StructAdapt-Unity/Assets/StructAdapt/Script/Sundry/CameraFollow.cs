using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public float cameraSpeed = 10f;
    public float rotationSpeed = 90f;
    public float verticalMoveSpeed = 5f;
    public float maxVerticalOffset = 10f;
    public float minVerticalOffset = 2f;

    private Vector3 offset;
    private Transform playerTransform;

    void Start()
    {
        playerTransform = GameObject.FindGameObjectWithTag("Player").transform;
        offset = transform.position - playerTransform.position;
    }

    void Update()
    {
        transform.position = playerTransform.position + offset;

        HandleMouseScroll();

        HandleHorizontalRotationWithWASD();

        HandleVerticalMovementWithWASD();
    }

    void HandleMouseScroll()
    {
        float scroll = -Input.GetAxis("Mouse ScrollWheel") * cameraSpeed;
        Camera.main.fieldOfView += scroll;
        Camera.main.fieldOfView = Mathf.Clamp(Camera.main.fieldOfView, 15, 60);
    }

    void HandleHorizontalRotationWithWASD()
    {
        float rotationAmount = 0f;

        if (Input.GetKey(KeyCode.A))
        {
            rotationAmount = rotationSpeed * Time.deltaTime;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            rotationAmount = -rotationSpeed * Time.deltaTime;
        }

        offset = Quaternion.Euler(0, rotationAmount, 0) * offset;
    }

    void HandleVerticalMovementWithWASD()
    {
        float verticalMovement = 0f;

        if (Input.GetKey(KeyCode.W))
        {
            verticalMovement = verticalMoveSpeed * Time.deltaTime;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            verticalMovement = -verticalMoveSpeed * Time.deltaTime;
        }

        offset += Vector3.up * verticalMovement;

        offset.y = Mathf.Clamp(offset.y, minVerticalOffset, maxVerticalOffset);
    }

    void LateUpdate()
    {
        transform.LookAt(playerTransform);
    }
}